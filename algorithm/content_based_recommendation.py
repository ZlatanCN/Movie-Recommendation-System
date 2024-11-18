import sys
import io
import json

from flask import jsonify, Flask
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import col, lit

from py4j.java_gateway import java_import
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import CountVectorizer

# 设置标准输出的编码格式为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)


# 路由1: 基于内容的推荐功能
@app.route('/api/recommendation/content/<int:movie_id>', methods=['GET'])
def content_based_recommendation(movie_id):
    def create_spark_session(memory="16g", max_executors=10):
        """
        创建并返回SparkSession

        :param max_executors: 整数，表示 Spark 作业的最大执行器数量，默认是 10。
        :param memory: 字符串，表示驱动程序内存配置，默认是"16g"。
        :return: 返回一个SparkSession对象，用于执行Spark作业。
        """
        return SparkSession.builder \
            .appName("ContentBasedMovieRecommendation") \
            .config("spark.driver.memory", memory) \
            .config("spark.executor.memory", memory) \
            .config("spark.executor.memoryOverhead", "2g") \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.dynamicAllocation.minExecutors", "1") \
            .config("spark.dynamicAllocation.maxExecutors", str(max_executors)) \
            .config("spark.sql.shuffle.partitions", "1000") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
            .getOrCreate()

    def load_and_clean_data(spark, file_path):
        """
        加载并清洗电影数据集

        :param spark: SparkSession对象，用于与Spark交互。
        :param file_path: 字符串，表示电影数据集的文件路径。
        :return: DataFrame，包含清洗后的电影数据集。
        """
        # 读取 CSV 文件，推断数据类型和包含表头
        movie_df = spark.read.csv(file_path, header=True, inferSchema=True)

        # 将 'vote_average' 列的数据类型转换为浮点型，并过滤掉评分为 0 的电影数据
        movie_df = movie_df.withColumn("vote_average", movie_df["vote_average"].cast(FloatType())).filter(
            "vote_average != 0")

        # 选择所需的字段，包括电影ID、标题、类型、评分、是否为成人电影和原始语言
        columns_to_keep = ['id', 'title', 'genres', 'vote_average', 'adult', 'original_language']
        movie_df = movie_df.select(columns_to_keep).na.fill({'genres': 'unknown'}).dropDuplicates()

        # 将清洗后的数据缓存以减少重复I/O操作
        movie_df = movie_df.cache()
        return movie_df

    def process_genres(spark, movie_df):
        """
        处理电影类型（genres）数据，创建并应用计数向量化

        :param spark: SparkSession对象。
        :param movie_df: DataFrame，包含电影数据。
        :return: DataFrame，添加了 'genres_vector' 列，表示类型的向量化数据。
        """
        # 将 'genres' 列的字符串分割成数组
        split_genres = F.split(movie_df.genres, ',')

        # 使用 CountVectorizer 对 'genres_array' 列进行词频编码（one-hot encoding）
        cv = CountVectorizer(inputCol="genres_array", outputCol="genres_vector")

        # 将类型（genres）字符串转换为小写并去除空格
        movie_df = movie_df.withColumn("genres_array", F.transform(split_genres, lambda x: F.trim(F.lower(x))))

        # 对电影类型数组进行词频编码，并将结果存储在 'genres_vector' 列中
        cv_model = cv.fit(movie_df)
        movie_df = cv_model.transform(movie_df)
        return movie_df

    def clean_titles_and_languages(movie_df):
        """
        清洗电影标题和原始语言字段，简化语言类别

        :param movie_df: DataFrame，包含电影数据。
        :return: DataFrame，包含清洗后的标题和简化的语言列。
        """
        # 清洗电影标题：去除标题空格，转为小写，去掉所有空格
        movie_df = movie_df \
            .withColumn('title', F.regexp_replace(F.lower(F.trim(F.col('title'))), ' ', '')) \
            .withColumn('original_language', F.regexp_replace(F.lower(F.trim(F.col('original_language'))), ' ', ''))

        # 定义主要语言列表，非主要语言归类为 "else"
        main_languages = ['en', 'cn', 'ja', 'kr']
        movie_df = movie_df.withColumn('original_language',
                                       F.when(F.col('original_language').isin(main_languages),
                                              F.col('original_language')).otherwise('else')).cache()
        return movie_df

    def process_categorical_features(movie_df):
        """
        对分类特征（是否为成人电影、原始语言）进行编码

        :param movie_df: DataFrame，包含电影数据。
        :return: DataFrame，包含编码后的特征列。
        """
        # 使用 StringIndexer 对成人标识（adult）和原始语言（original_language）进行索引编码
        indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") for col in
                    ["adult", "original_language"]]

        # 对索引结果进行 one-hot 编码
        encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec", handleInvalid="keep") for col in
                    ["adult", "original_language"]]

        # 创建并执行编码 Pipeline，返回编码后的 DataFrame
        pipeline = Pipeline(stages=indexers + encoders)
        movie_df = pipeline.fit(movie_df).transform(movie_df)
        return movie_df

    def normalize_features(movie_df):
        """
        标准化数值特征，包括评分、类型向量、是否成人和原始语言

        :param movie_df: DataFrame，包含电影数据。
        :return: DataFrame，包含标准化后的特征。
        """
        # 使用 VectorAssembler 合并所有数值特征
        assembler = VectorAssembler(inputCols=["vote_average", "genres_vector", "adult_vec", "original_language_vec"],
                                    outputCol="features")

        # 使用 StandardScaler 对特征进行标准化
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

        # 创建并执行标准化 Pipeline
        pipeline = Pipeline(stages=[assembler, scaler])
        movie_df = pipeline.fit(movie_df).transform(movie_df)
        return movie_df

    def calculate_similarities(movie_df, target_id):
        """
        计算目标电影与所有其他电影的相似度，返回相似度最高的20部电影

        :param movie_df: DataFrame，包含电影数据。
        :param target_id: 目标电影ID。
        :return: DataFrame，包含相似度最高的20部电影信息。
        """
        # 获取目标电影的特征向量
        target_movie = movie_df.filter(F.col("id") == target_id).first()
        if not target_movie:
            raise ValueError(f"Movie with ID {target_id} not found")
        target_features = target_movie.scaled_features

        # 定义计算余弦相似度的函数
        def cosine_similarity(v1, v2):
            return float(v1.dot(v2) / (Vectors.norm(v1, 2) * Vectors.norm(v2, 2)))

        similarity_udf = F.udf(lambda x: cosine_similarity(x, target_features), FloatType())

        # 计算所有电影与目标电影的相似度，按相似度排序并选出前20名
        similarities = movie_df.filter(F.col("id") != target_id) \
            .withColumn("similarity", similarity_udf(F.col("scaled_features"))) \
            .select("id", "title", "similarity") \
            .orderBy(F.desc("similarity")) \
            .limit(20)
        return similarities

    def load_processed_data(spark, file_path):
        """
        加载处理后的数据

        :param spark: SparkSession对象。
        :param file_path: 字符串，表示处理后的数据路径。
        :return: DataFrame，包含处理后的数据。
        """
        return spark.read.parquet(file_path)

    def is_path_exists(spark, path):
        """
        检查指定的 HDFS 路径是否存在

        :param spark: SparkSession对象。
        :param path: 字符串，表示 HDFS 路径。
        :return: 布尔值，True 表示路径存在，False 表示路径不存在。
        """
        # 使用 Java API 检查指定路径是否存在
        sc = spark.sparkContext
        java_import(sc._gateway.jvm, "org.apache.hadoop.fs.FileSystem")
        java_import(sc._gateway.jvm, "org.apache.hadoop.fs.Path")
        fs = sc._jvm.FileSystem.get(sc._jsc.hadoopConfiguration())
        return fs.exists(sc._jvm.Path(path))

    spark = create_spark_session()
    try:
        initial_dataset_path = '/TMDB_dataset/TMDB_movie_dataset_v11.csv'
        processed_data_path = '/TMDB_dataset/processed_movie_data'

        # 检查处理后的数据路径是否存在
        if not is_path_exists(spark, processed_data_path):
            # 如果路径不存在，加载和处理原始数据
            movie_df = load_and_clean_data(spark, initial_dataset_path)
            movie_df = process_genres(spark, movie_df)
            movie_df = clean_titles_and_languages(movie_df)
            movie_df = process_categorical_features(movie_df)
            movie_df = normalize_features(movie_df)
            # 将处理后的数据存储到 HDFS
            movie_df.write.mode("overwrite").parquet(processed_data_path)
        else:
            # 如果路径存在，直接加载处理后的数据
            movie_df = load_processed_data(spark, processed_data_path)

        if movie_id is None:
            movie_id = 238
        recommendations = calculate_similarities(movie_df, movie_id)

        # 将推荐结果转换为 JSON 格式并输出
        result = {str(row.id): float(row.similarity) for row in recommendations.collect()}
        print(json.dumps(result))
        return jsonify({"isSuccessful": True, "content": result})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"isSuccessful": False, "message": str(e)})
    finally:
        spark.stop()


# 路由2: 基于协同过滤 (ALS) 的推荐功能
@app.route('/api/recommendation/collaborative/<int:user_id>', methods=['GET'])
def collaborative_filtering(user_id):
    """
    基于用户协同过滤的电影推荐功能。
    此方法根据用户 ID，使用 ALS 模型生成电影推荐列表。

    :param user_id: 用户 ID (整数类型)。
    :return: 包含推荐结果的 JSON 响应。
    """

    def create_spark_session(memory="16g", max_executors=10):
        """
        创建并返回一个配置好的 SparkSession 实例。

        :param memory: 分配给 Spark driver 和 executor 的内存大小，默认是 "16g"。
        :param max_executors: Spark 动态分配的最大 executor 数量，默认是 10。
        :return: 配置完成的 SparkSession 对象。
        """
        return SparkSession.builder \
            .appName("ALSMovieRecommendation") \
            .config("spark.driver.memory", memory) \
            .config("spark.executor.memory", memory) \
            .config("spark.executor.instances", "20") \
            .config("spark.executor.cores", "4") \
            .config("spark.rdd.compress", "true") \
            .config("spark.network.timeout", "1000s") \
            .config("spark.sql.broadcastTimeout", "800") \
            .config("spark.executor.heartbeatInterval", "100s") \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.dynamicAllocation.maxExecutors", str(max_executors)) \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.memory.storageFraction", "0.3") \
            .getOrCreate()

    def load_and_clean_data(spark, file_path, sample_fraction=0.3):
        """
        加载并清洗评分数据和电影数据。

        :param spark: SparkSession 对象。
        :param file_path: 包含电影数据和评分数据的文件路径列表。
        :param sample_fraction: 采样比例，默认是 0.3，代表只加载部分数据。
        :return: 清洗后的评分数据 PySpark DataFrame。
        """
        # 加载电影数据和评分数据
        movies = spark.read.csv(file_path[0], header=True, inferSchema=True)
        ratings = spark.read.csv(file_path[1], header=True, inferSchema=True)

        # 清理评分数据
        ratings = clean_ratings(ratings)

        # 从电影数据中选择 id 和 title 两列，并与评分数据按 tmdbId 进行内连接
        movies = movies.select("id", "title")
        ratings = ratings.join(movies, ratings.tmdbId == movies.id, "inner").drop("id")

        return ratings

    def clean_ratings(data):
        """
        清理评分数据。
        删除时间戳、movieId 列，并移除包含 null 值的记录。

        :param data: 原始的 PySpark DataFrame。
        :return: 清理后的 PySpark DataFrame。
        """
        data = data.drop('timestamp')
        data = data.drop('movieId')
        data = data.dropna()
        return data

    def RMSE(predictions):
        """
        计算预测数据的均方根误差 (Root Mean Squared Error, RMSE)。

        :param predictions: 包含真实评分和预测评分的 PySpark DataFrame。
        :return: 浮点数，表示 RMSE 值。
        """
        # 过滤掉含有 null 值的记录
        predictions = predictions.filter(col("rating").isNotNull() & col("prediction").isNotNull())

        # 如果 predictions 数据为空，直接返回 0.0
        if predictions.count() == 0:
            return 0.0

        # 计算平方误差并取均值
        squared_diff = predictions.withColumn("squared_diff", pow(col("rating") - col("prediction"), 2))
        mse_row = squared_diff.selectExpr("mean(squared_diff) as mse").first()
        mse = mse_row.mse if mse_row.mse is not None else 0.0

        # 返回 RMSE
        return mse ** 0.5

    def train_model(data):
        """
        训练 ALS 模型并返回最佳模型。

        :param data: PySpark DataFrame，包含评分数据。
        :return: 训练完成的 ALS 模型。
        """
        # 将数据随机分为训练集、验证集和测试集
        train, validation, test = data.randomSplit([0.6, 0.2, 0.2], seed=0)
        print("训练集、验证集、测试集的数据量分别为: {}, {}, {}".format(train.count(), validation.count(), test.count()))

        def grid_search(train, validation, num_iterations, reg_params, ranks):
            """
            使用网格搜索优化 ALS 模型。

            :param train: 训练集 PySpark DataFrame。
            :param validation: 验证集 PySpark DataFrame。
            :param num_iterations: ALS 迭代次数列表。
            :param reg_params: 正则化参数列表。
            :param ranks: 潜在因子数量列表。
            :return: 最优的 ALS 模型。
            """
            min_error = float('inf')  # 最小 RMSE
            best_model = None  # 最优模型

            for rank in ranks:
                for reg in reg_params:
                    als = ALS(
                        rank=rank,
                        maxIter=num_iterations,
                        seed=0,
                        regParam=reg,
                        userCol="userId",
                        itemCol="tmdbId",
                        ratingCol="rating",
                        coldStartStrategy="drop"
                    )
                    model = als.fit(train)
                    predictions = model.transform(validation)

                    # 计算验证集的 RMSE
                    rmse = RMSE(predictions)
                    print(f'{rank} 个潜在因子, 正则化参数为 {reg}: 验证集 RMSE 为 {rmse}')

                    if rmse < min_error:
                        min_error = rmse
                        best_model = model

            print(f'\n最优模型潜在因子数: {rank}, 正则化参数: {reg}')
            return best_model

        # 网格搜索参数
        best_model = grid_search(
            train,
            validation,
            num_iterations=10,
            reg_params=[0.05, 0.1, 0.2, 0.4, 0.8],
            ranks=[6, 8, 10, 12]
        )

        return best_model

    def predict_for_user(user_id, model, ratings, top_n=10):
        """
        为指定用户生成电影推荐列表。

        :param user_id: 用户 ID (整数)。
        :param model: 已训练好的 ALS 模型。
        :param ratings: PySpark DataFrame，包含评分数据。
        :param top_n: 推荐的电影数量，默认为 10。
        :return: 包含推荐电影的 PySpark DataFrame。
        """

        def get_unrated_items(user_id, ratings):
            """
            获取指定用户未评分的电影。
            :param user_id: 用户 ID。
            :param ratings: PySpark DataFrame，包含评分数据。
            :return: 包含未评分电影的 PySpark DataFrame。
            """
            rated_items = ratings.filter(col("userId") == user_id).select("tmdbId")
            unrated_items = ratings.select("tmdbId", "title").distinct().join(
                rated_items, "tmdbId", "left_anti"
            )
            return unrated_items

        def generate_user_item_pairs(user_id, unrated_items):
            """
            为指定用户生成所有未评分电影的用户-电影对。
            :param user_id: 用户 ID。
            :param unrated_items: 未评分电影的 PySpark DataFrame。
            :return: 包含用户-电影对的 PySpark DataFrame。
            """
            user_df = spark.createDataFrame([(user_id,)], ["userId"])
            user_item_pairs = user_df.crossJoin(unrated_items)
            return user_item_pairs

        # 获取用户未评分的物品
        unrated_items = get_unrated_items(user_id, ratings)

        # 为用户生成用户-物品对
        user_item_pairs = generate_user_item_pairs(user_id, unrated_items)

        # 分区处理以提高预测效率
        user_item_pairs = user_item_pairs.repartition(100, "userId")

        # 使用模型预测评分
        predictions = model.transform(user_item_pairs)

        # 过滤掉无效预测并获取前 top_n 推荐
        top_predictions = predictions.filter(col("prediction").isNotNull()) \
            .orderBy(col("prediction").desc()) \
            .limit(top_n)

        # 根据预测评分降序排序并取前 top_n 个结果
        return top_predictions.select("userId", "tmdbId", "title", "prediction")

    def is_model_exists(spark, path):
        """
        检查 ALS 模型是否存在于指定的路径中。

        该函数通过 Spark 的 Hadoop 文件系统 API 检查指定路径是否存在，
        以判断是否已经保存了训练好的 ALS 模型。

        :param spark: SparkSession 对象，用于与 Spark 环境交互。
        :param path: 字符串，表示保存 ALS 模型的路径。
        :return: 布尔值，若模型存在返回 True，否则返回 False。
        """
        # 使用 Spark 的 JVM API 获取 Hadoop 文件系统的访问对象
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        # 检查路径是否存在
        return fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))

    def get_table_from_mysql(spark, table_name, mysql_url, mysql_user, mysql_password):
        """
        从 MySQL 数据库中读取指定表的数据，并返回 PySpark DataFrame。

        通过 PySpark 的 JDBC 接口从 MySQL 数据库加载表数据，支持大规模数据读取，
        并提供批量大小和并行性配置以优化性能。

        :param spark: SparkSession 对象，用于与 Spark 环境交互。
        :param table_name: 字符串，表示 MySQL 数据库中的表名。
        :param mysql_url: 字符串，表示 MySQL 数据库的 JDBC URL。
        :param mysql_user: 字符串，表示 MySQL 数据库的用户名。
        :param mysql_password: 字符串，表示 MySQL 数据库的密码。
        :return: PySpark DataFrame，包含从指定 MySQL 表中读取的数据。
        """
        # 使用 PySpark 的 read 方法通过 JDBC 接口读取 MySQL 表数据
        data = spark.read \
            .format("jdbc") \
            .option("url", mysql_url) \
            .option("dbtable", table_name) \
            .option("user", mysql_user) \
            .option("password", mysql_password) \
            .option("fetchsize", "10000") \
            .option("useCompression", "true") \
            .load()

        # 返回读取到的 DataFrame
        return data

    # 初始化 Spark 会话
    spark = create_spark_session()

    # 配置数据路径
    ratings_path = "/TMDB_dataset/ratings_with_tmdbid.csv"
    model_path = "/TMDB_dataset/als_model"

    try:
        # 从 MySQL 加载评分数据
        mysql_url = "jdbc:mysql://localhost:3307/movie_recommendation"
        mysql_user = "root"
        mysql_password = "zhujiayou"
        ratings = get_table_from_mysql(spark, "ratings", mysql_url, mysql_user, mysql_password)

        # 检查模型是否存在
        if is_model_exists(spark, model_path):
            model = ALSModel.load(model_path)  # 加载现有模型
        else:
            model = train_model(ratings)  # 训练新模型
            model.save(model_path)  # 保存模型

        # 为指定用户生成推荐列表
        top_predictions = predict_for_user(user_id, model, ratings)

        # 转换为 JSON 格式
        top_predictions = {str(row.tmdbId): float(row.prediction) for row in top_predictions.collect()}
        print(json.dumps(top_predictions))
        return jsonify({
            "isSuccessful": True,
            "content": top_predictions
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"isSuccessful": False, "message": str(e)})
    finally:
        spark.stop()


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=6000)
