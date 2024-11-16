import sys
import io
import json

from flask import Flask, request, jsonify

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

initial_dataset_path = '/TMDB_dataset/TMDB_movie_dataset_v11.csv'
processed_data_path = '/TMDB_dataset/processed_movie_data.parquet'
movie_table_name = 'default.tmdb_movie'

# 将 initial_dataset_path 的数据加载到 Hive 表 movie_table_name 中
spark = SparkSession.builder \
    .appName("Test") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

print("Is Hive enabled:", spark.conf.get("spark.sql.catalogImplementation"))

# 读取 CSV 文件
df = spark.read.csv(initial_dataset_path, header=True, inferSchema=True)

# 将数据写入 Hive 表
df.write.mode("overwrite").saveAsTable(movie_table_name)

spark.stop()