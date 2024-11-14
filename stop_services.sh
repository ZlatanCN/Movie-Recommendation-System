#!/bin/bash

# 关闭 Hadoop
/usr/local/hadoop/sbin/stop-all.sh

# 关闭 Spark
/usr/local/spark/sbin/stop-all.sh

echo "All services stopped."