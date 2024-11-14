import sys
import io
import json

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


def create_spark_session(memory="16g", max_executors=10):
    """
    创建并返回SparkSession

    :param max_executors: 整数，表示 Spark 作业的最大执行器数量，默认是 10。
    :param memory: 字符串，表示驱动程序内存配置，默认是"16g"。
    :return: 返回一个SparkSession对象，用于执行Spark作业。
    """
    return SparkSession.builder \
        .appName("MovieRecommendation") \
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


def main():
    """
    主函数，执行数据加载、处理和相似度计算流程
    """
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

        # 从命令行读取目标电影 ID（默认为 424）
        movie_id = int(sys.argv[1]) if len(sys.argv) > 1 else 424
        recommendations = calculate_similarities(movie_df, movie_id)

        # 将推荐结果转换为 JSON 格式并输出
        result = {str(row.id): float(row.similarity) for row in recommendations.collect()}
        print(json.dumps(result))

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
