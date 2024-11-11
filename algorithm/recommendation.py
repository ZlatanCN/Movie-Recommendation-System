from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, expr, lit
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

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

def evaluate_model(predictions):
    """
    Evaluate the model using RMSE, MSE, MAE, and R2 metrics.

    :param predictions: A PySpark DataFrame
    :return: None
    """

    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator_rmse.evaluate(predictions)

    evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
    mse = evaluator_mse.evaluate(predictions)

    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
    mae = evaluator_mae.evaluate(predictions)

    evaluator_r2 = RegressionEvaluator(metricName="r2", labelCol="rating", predictionCol="prediction")
    r2 = evaluator_r2.evaluate(predictions)

    # Create a DataFrame containing all the metrics
    schema = StructType([
        StructField("Metric", StringType(), True),
        StructField("Value", FloatType(), True)
    ])

    metrics_df = spark.createDataFrame([
        ("RMSE", rmse),
        ("MSE", mse),
        ("MAE", mae),
        ("R2", r2)
    ], schema)

    # Display the results
    metrics_df.show()

def predict_for_user(user_id, model, ratings, top_n=10):
    """
    Make movie recommendations for a specific user.

    :param user_id: An integer representing the user ID
    :param model: A trained ALS model
    :param ratings: A PySpark DataFrame containing the ratings data
    :param top_n: An integer representing the number of recommendations to return
    :return: A PySpark DataFrame containing the top N movie recommendations for the user
    """

    # 获取所有物品的 tmdbId 和标题
    all_items = ratings.select("tmdbId", "title").distinct()

    # 创建用户-物品对的 DataFrame，用户 ID 设置为指定的 user_id
    user_item_pairs = all_items.withColumn("userId", lit(user_id))

    # 进行预测
    predictions = model.transform(user_item_pairs)

    # 根据预测评分降序排列，取前 top_n 个推荐结果
    top_predictions = predictions.orderBy(col("prediction").desc()).limit(top_n)

    # 显示结果
    top_predictions.select("userId", "tmdbId", "title", "prediction").show()

    return top_predictions

# 初始化 Spark 会话，配置更多的内存和其他相关参数
spark = SparkSession.builder \
    .appName("RecommendationSystem") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.5") \
    .getOrCreate()

# ratings_with_tmdbid.csv 是一个包含用户对电影的评分数据集，从 HDFS 中读取
# TMDB_movie_dataset_v11.csv 是一个包含电影信息的数据集，从 HDFS 中读取
ratings = spark.read.csv("/TMDB_dataset/ratings_with_tmdbid.csv", header=True, inferSchema=True)
movies = spark.read.csv("/TMDB_dataset/TMDB_movie_dataset_v11.csv", header=True, inferSchema=True)

try:
    ratings = clean_ratings(ratings)
    ratings = ratings.sample(fraction=0.3, withReplacement=False) # 采样 30% 的数据

    # 从电影信息数据集中选择 'id' 和 'title' 列, 并将其与评分数据集进行内连接
    movies = movies.select("id", "title")
    ratings = ratings.join(movies, ratings.tmdbId == movies.id, "inner").drop("id")

    # 将数据集分为训练集和测试集
    train, test = ratings.randomSplit([0.8, 0.2])

    # 使用 ALS 算法训练模型
    als = ALS(maxIter=3, regParam=0.05, userCol="userId", itemCol="tmdbId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(train)

    # 使用测试集评估模型
    test_predictions = model.transform(test)
    evaluate_model(test_predictions)

    # 为用户 1 生成推荐结果
    predict_for_user(1, model, ratings)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    spark.stop()