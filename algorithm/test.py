import sys
import io
import json
import pymongo
from bson.binary import Binary
import pickle
from pyexpat import features

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import CountVectorizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def create_spark_session(memory="4g"):
    """
    创建并返回SparkSession
    :param memory: 字符串，表示内存配置
    """
    return SparkSession.builder \
        .appName("MovieRecommendation") \
        .config("spark.driver.memory", memory) \
        .getOrCreate()


def load_and_clean_data(spark, file_path):
    """
    加载并清洗电影数据集
    :param spark: SparkSession对象
    :param file_path: 字符串，表示电影数据集的路径
    :return: 返回一个 DataFrame，包含清洗后的电影数据集
    """
    # 读取CSV文件
    movie_df = spark.read.csv(file_path, header=True, inferSchema=True)

    # 过滤评分为0的电影
    movie_df = movie_df.withColumn("vote_average", movie_df["vote_average"].cast(FloatType()))
    movie_df = movie_df.filter(movie_df.vote_average != 0)

    # 选择需要的列
    columns_to_keep = ['id', 'title', 'genres', 'vote_average', 'adult', 'original_language']
    movie_df = movie_df.select(columns_to_keep)

    # 填充空值
    movie_df = movie_df.na.fill({'genres': 'unknown'})

    # 去重
    movie_df = movie_df.dropDuplicates()

    return movie_df


def process_genres(spark, movie_df):
    """
    处理电影类型数据
    :param spark: SparkSession对象
    :param movie_df: DataFrame，包含电影数据
    :return: 返回处理后的DataFrame
    """
    # 分割genres字符串为数组
    split_genres = F.split(movie_df.genres, ',')

    # 创建CountVectorizer来处理genres
    cv = CountVectorizer(inputCol="genres_array", outputCol="genres_vector")

    # 将genres转换为数组格式
    movie_df = movie_df.withColumn("genres_array",
                                   F.transform(split_genres, lambda x: F.trim(F.lower(x))))

    # 拟合并转换genres
    cv_model = cv.fit(movie_df)
    movie_df = cv_model.transform(movie_df)

    return movie_df


def clean_titles_and_languages(movie_df):
    """
    清洗电影数据
    :param movie_df: DataFrame，包含电影数据
    :return: 返回清洗后的 DataFrame
    """
    # 清洗电影标题：去除空格、转为小写字母并去除所有空格
    movie_df = movie_df.withColumn('title', F.lower(F.trim(F.col('title'))).alias('title'))
    movie_df = movie_df.withColumn('title', F.regexp_replace('title', ' ', ''))

    # 清洗电影原始语言：去除空格、转为小写字母并去除所有空格
    movie_df = movie_df.withColumn('original_language', F.lower(F.trim(F.col('original_language'))))
    movie_df = movie_df.withColumn('original_language', F.regexp_replace('original_language', ' ', ''))

    # 将非主要语言归类为“else”
    main_languages = ['en', 'cn', 'ja', 'kr']
    movie_df = movie_df.withColumn('original_language',
                                   F.when(F.col('original_language').isin(main_languages), F.col('original_language'))
                                   .otherwise('else'))
    return movie_df


def process_categorical_features(movie_df):
    """
    处理分类特征
    :param movie_df: DataFrame，包含电影数据
    :return: 返回处理后的DataFrame
    """
    # 处理adult和original_language列
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip")
        for col in ["adult", "original_language"]
    ]

    encoders = [
        OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec", handleInvalid="keep")
        for col in ["adult", "original_language"]
    ]

    # 创建并执行pipeline
    pipeline = Pipeline(stages=indexers + encoders)
    movie_df = pipeline.fit(movie_df).transform(movie_df)

    return movie_df


def normalize_features(movie_df):
    """
    标准化数值特征
    :param movie_df: DataFrame，包含电影数据
    :return: 返回标准化后的DataFrame
    """
    # 组合所有特征
    assembler = VectorAssembler(
        inputCols=["vote_average", "genres_vector", "adult_vec", "original_language_vec"],
        outputCol="features"
    )

    # 标准化
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    # 创建并执行pipeline
    pipeline = Pipeline(stages=[assembler, scaler])
    movie_df = pipeline.fit(movie_df).transform(movie_df)

    return movie_df


def calculate_similarities(movie_df, target_id):
    """
    计算电影相似度
    :param movie_df: DataFrame，包含电影数据
    :param target_id: 目标电影ID
    :return: 返回相似度最高的20部电影
    """
    # 获取目标电影的特征
    target_movie = movie_df.filter(F.col("id") == target_id).first()
    if not target_movie:
        raise ValueError(f"Movie with ID {target_id} not found")

    target_features = target_movie.scaled_features

    # 计算余弦相似度
    def cosine_similarity(v1, v2):
        return float(v1.dot(v2) / (Vectors.norm(v1, 2) * Vectors.norm(v2, 2)))

    similarity_udf = F.udf(lambda x: cosine_similarity(x, target_features), FloatType())

    # 计算所有电影与目标电影的相似度
    similarities = movie_df.filter(F.col("id") != target_id) \
        .withColumn("similarity", similarity_udf(F.col("scaled_features"))) \
        .select("id", "title", "similarity") \
        .orderBy(F.desc("similarity")) \
        .limit(20)

    return similarities


def main():
    spark = create_spark_session()

    try:
        # 获取命令行参数
        movie_id = int(sys.argv[1]) if len(sys.argv) > 1 else 424

        # 加载和处理数据
        movie_df = load_and_clean_data(spark, '/TMDB_dataset/TMDB_movie_dataset_v11.csv') # 在 HDFS 上的路径
        movie_df = process_genres(spark, movie_df)
        movie_df = clean_titles_and_languages(movie_df)
        movie_df = process_categorical_features(movie_df)
        movie_df = normalize_features(movie_df)

        # 获取推荐
        recommendations = calculate_similarities(movie_df, movie_id)

        # 转换结果为JSON格式
        result = {str(row.id): float(row.similarity) for row in recommendations.collect()}
        print(json.dumps(result))

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
