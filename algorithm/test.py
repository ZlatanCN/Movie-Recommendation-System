import sys
import io
import json
import time
import logging

from flask import jsonify, Flask
from mpmath import limit
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collaborative_filtering(user_id):
    def create_spark_session(memory="16g", max_executors=10):
        """
        Create and return a SparkSession.

        :param memory: A string representing the memory configuration for the driver, default is "16g".
        :param max_executors: An integer representing the maximum number of executors for Spark jobs, default is 10.
        :return: A SparkSession object for executing Spark jobs.
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
        Load and clean the movie data.
        :param sample_fraction: A float representing the fraction of the data to sample.
        :param spark: A SparkSession object for interacting with Spark.
        :param file_path: An array of strings representing the file paths of the movie and ratings data.
        :return: A PySpark DataFrame containing the cleaned movie data.
        """

        movies = spark.read.csv(file_path[0], header=True, inferSchema=True)
        ratings = spark.read.csv(file_path[1], header=True, inferSchema=True)

        ratings = clean_ratings(ratings)
        # ratings = ratings.sample(fraction=sample_fraction, withReplacement=False)

        # 从电影信息数据集中选择 'id' 和 'title' 列, 并将其与评分数据集进行内连接
        movies = movies.select("id", "title")
        ratings = ratings.join(movies, ratings.tmdbId == movies.id, "inner").drop("id")

        return ratings

    def print_missing_values(data):
        """
        Print the number of missing values in each column of a DataFrame.

        :param data: A PySpark DataFrame
        :return: None
        """

        for column in data.columns:
            print(column + ": " + str(data.filter(col(column).isNull()).count()))

        print("\n")

    def clean_ratings(data):
        """
        Drop the 'timestamp' and 'movieId' columns,
        and remove rows with missing values in 'userId', 'tmdbId',
        and 'rating' columns.

        :param data: A PySpark DataFrame
        :return: A PySpark DataFrame
        """

        # print("Number of missing values before drop and dropna: ")
        # print_missing_values(data)

        data = data.drop('timestamp')
        data = data.drop('movieId')
        data = data.dropna()

        # print("Number of missing values after drop and dropna: ")
        # print_missing_values(data)

        return data

    def RMSE(predictions):
        """
        Calculate the Root Mean Squared Error (RMSE) of the predictions.

        :param predictions: A PySpark DataFrame containing the predictions
        :return: A float representing the RMSE
        """

        # 检查并过滤 NULL 值
        predictions = predictions.filter(col("rating").isNotNull() & col("prediction").isNotNull())

        # 如果 predictions 为空，直接返回 0.0
        if predictions.count() == 0:
            return 0.0

        # 计算平方误差
        squared_diff = predictions.withColumn("squared_diff", pow(col("rating") - col("prediction"), 2))

        # 计算均方误差（MSE）
        mse_row = squared_diff.selectExpr("mean(squared_diff) as mse").first()
        mse = mse_row.mse if mse_row.mse is not None else 0.0

        # 返回 RMSE
        return mse ** 0.5

    def train_model(data):
        """
        Train an ALS model using the given data.

        :param spark: A SparkSession object
        :param data: A PySpark DataFrame
        :return: A trained ALS model
        """

        train, validation, test = data.randomSplit([0.6, 0.2, 0.2], seed=0)
        print("The number of ratings in each set: {}, {}, {}".format(train.count(), validation.count(), test.count()))

        def grid_search(train, validation, num_iterations, reg_params, ranks):
            """
            Perform grid search to find the best ALS model.

            :param train: A PySpark DataFrame containing the training data
            :param validation: A PySpark DataFrame containing the validation data
            :param num_iterations: A list of integers representing the number of iterations to test
            :param reg_params: A list of floats representing the regularization parameters to test
            :param ranks: A list of integers representing the ranks to test
            :return: The best ALS model
            """

            min_error = float('inf')
            best_rank = -1
            best_regularization = 0
            best_model = None

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

                    rmse = RMSE(predictions)
                    print('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, rmse))

                    if rmse < min_error:
                        min_error = rmse
                        best_rank = rank
                        best_regularization = reg
                        best_model = model

            pred_train = best_model.transform(train)
            train_error = RMSE(pred_train)

            print('\nThe best model has {} latent factors and regularization = {}:'.format(best_rank,
                                                                                           best_regularization))
            print('train RMSE is {}; validation RMSE is {}'.format(train_error, min_error))

            return best_model

        # Perform grid search to find the best ALS model
        start_time = time.time()
        best_model = grid_search(
            train,
            validation,
            10,
            [0.05, 0.1, 0.2, 0.4, 0.8],
            [6, 8, 10, 12]
        )
        print('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))

        # Evaluate the model using the test set
        pred_test = best_model.transform(test)
        test_error = RMSE(pred_test)
        print('The test RMSE is {}'.format(test_error))

        return best_model

    def predict_for_user(user_id, model, ratings, top_n=10):
        """
        为指定用户生成电影推荐列表。
        """

        def get_unrated_items(user_id, ratings):
            """
            获取用户未评分的物品列表。
            """
            rated_items = ratings.filter(col("userId") == user_id).select("tmdbId")
            unrated_items = ratings.select("tmdbId", "title").distinct().join(
                rated_items, "tmdbId", "left_anti"
            )
            return unrated_items

        def generate_user_item_pairs(user_id, unrated_items):
            """
            为指定用户生成用户-物品对。
            """
            user_df = spark.createDataFrame([(user_id,)], ["userId"])
            user_item_pairs = user_df.crossJoin(unrated_items)
            return user_item_pairs


        # 获取用户未评分的物品
        get_unrated_time = time.time()
        unrated_items = get_unrated_items(user_id, ratings)
        print("Unrated items retrieved successfully. Time taken: {:.2f} seconds".format(time.time() - get_unrated_time))

        # 为用户生成用户-物品对
        generate_pairs_time = time.time()
        user_item_pairs = generate_user_item_pairs(user_id, unrated_items)
        print("User-item pairs generated successfully. Time taken: {:.2f} seconds".format(time.time() - generate_pairs_time))

        # 分区处理以提高预测效率
        repartition_time = time.time()
        user_item_pairs = user_item_pairs.repartition(100, "userId")
        print("Data repartitioned successfully. Time taken: {:.2f} seconds".format(time.time() - repartition_time))

        # 使用模型进行预测
        transform_time = time.time()
        predictions = model.transform(user_item_pairs)
        print("Predictions generated successfully. Time taken: {:.2f} seconds".format(time.time() - transform_time))

        # 过滤掉无效预测并获取前 top_n 推荐
        filter_time = time.time()
        top_predictions = predictions.filter(col("prediction").isNotNull()) \
            .orderBy(col("prediction").desc()) \
            .limit(top_n)
        print("Top predictions filtered successfully. Time taken: {:.2f} seconds".format(time.time() - filter_time))

        # 返回最终推荐结果
        return top_predictions.select("userId", "tmdbId", "title", "prediction")


    def save_to_mysql(df, table_name, mysql_url, mysql_user, mysql_password):
        """
        Save the ratings data to a MySQL database.
        :param df: A PySpark DataFrame containing the ratings data.
        :param table_name: A string representing the name of the table in the MySQL database.
        :param mysql_url: A string representing the URL of the MySQL database.
        :param mysql_user: A string representing the username of the MySQL database.
        :param mysql_password: A string representing the password of the MySQL database.
        :return:
        """

        df.write \
            .format("jdbc") \
            .option("url", mysql_url) \
            .option("dbtable", table_name) \
            .option("user", mysql_user) \
            .option("password", mysql_password) \
            .option("batchsize", "5000") \
            .option("numPartitions", "10") \
            .mode("overwrite") \
            .save()

    def get_table_from_mysql(spark, table_name, mysql_url, mysql_user, mysql_password):
        """
        Get a table from a MySQL database.

        :param spark: A SparkSession object.
        :param table_name: A string representing the name of the table in the MySQL database.
        :param mysql_url: A string representing the URL of the MySQL database.
        :param mysql_user: A string representing the username of the MySQL database.
        :param mysql_password: A string representing the password of the MySQL database.
        :return: A PySpark DataFrame containing the data from the MySQL table.
        """

        data = spark.read \
            .format("jdbc") \
            .option("url", mysql_url) \
            .option("dbtable", table_name) \
            .option("user", mysql_user) \
            .option("password", mysql_password) \
            .option("fetchsize", "10000") \
            .load()

        return data

    def get_table_row_count(spark, table_name, mysql_url, mysql_user, mysql_password):
        """
        Get the number of rows in a MySQL table.

        :param spark: A SparkSession object.
        :param table_name: A string representing the name of the table in the MySQL database.
        :param mysql_url: A string representing the URL of the MySQL database.
        :param mysql_user: A string representing the username of the MySQL database.
        :param mysql_password: A string representing the password of the MySQL database.
        :return: An integer representing the number of rows in the table.
        """

        # Read the data from the MySQL table
        data = spark.read \
            .format("jdbc") \
            .option("url", mysql_url) \
            .option("dbtable", table_name) \
            .option("user", mysql_user) \
            .option("password", mysql_password) \
            .load()

        # Get the number of rows
        row_count = data.count()

        return row_count

    def is_model_exists(spark, path):
        """
        Check if the ALS model exists in the specified path.
        :param path: A string representing the path to the ALS model.
        :return: A boolean value indicating whether the model exists.
        """
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        return fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))

    # 初始化 Spark 会话，配置更多的内存和其他相关参数
    spark = create_spark_session()

    # ratings_with_tmdbid.csv 是一个包含用户对电影的评分数据集，从 HDFS 中读取
    # TMDB_movie_dataset_v11.csv 是一个包含电影信息的数据集，从 HDFS 中读取
    ratings_path = "/TMDB_dataset/ratings_with_tmdbid.csv"
    movies_path = "/TMDB_dataset/TMDB_movie_dataset_v11.csv"
    model_path = "/TMDB_dataset/als_model"

    try:
        # 加载和清洗数据
        # ratings = load_and_clean_data(spark, [movies_path, ratings_path])

        # 存储到 MySQL
        mysql_url = "jdbc:mysql://localhost:3307/movie_recommendation"
        mysql_user = "root"
        mysql_password = "zhujiayou"
        # save_to_mysql(ratings, "ratings", mysql_url, mysql_user, mysql_password)

        # print("Data saved to MySQL successfully.")

        load_data_time = time.time()
        ratings = get_table_from_mysql(spark, "ratings", mysql_url, mysql_user, mysql_password)
        print("Data loaded from MySQL successfully. Time taken: {:.2f} seconds".format(time.time() - load_data_time))

        if is_model_exists(spark, model_path):
            model = ALSModel.load(model_path)
            logger.info("Model loaded successfully.")
        else:
            # 对数据进行分区和缓存
            ratings = ratings.repartition(50).cache()
            # 训练 ALS 模型
            model = train_model(ratings)
            logger.info("Model training completed successfully.")
            # 保存模型
            model.save(model_path)
            logger.info("Model saved successfully.")

        #
        # 为用户 1 生成推荐结果

        predict_time = time.time()
        top_predictions = predict_for_user(1, model, ratings)
        print("Recommendations generated successfully. Time taken: {:.2f} seconds".format(time.time() - predict_time))
        top_predictions.show()

        logger.info("Recommendations generated successfully.")

        #
        # # 将推荐结果转换为 JSON 格式并输出
        # top_predictions = {str(row.tmdbId): float(row.prediction) for row in top_predictions.collect()}
        # print(top_predictions)
        # # return jsonify({
        # #     "isSuccessful": True,
        # #     "content": top_predictions
        # # })

    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error("Error during model training: %s", str(e))
    finally:
        spark.stop()


if __name__ == '__main__':
    collaborative_filtering(1)
