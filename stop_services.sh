#!/bin/bash

# 关闭 Hadoop
/usr/local/hadoop/sbin/stop-all.sh

# 关闭 Spark
/usr/local/spark/sbin/stop-all.sh

# 关闭 Flask 监听 content-based 推荐请求
pkill -f /mnt/c/Users/hasee/Desktop/movie-recommendation/algorithm/content_based_recommendation.py

echo "All services stopped."