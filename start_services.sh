#!/bin/bash

# 确保脚本在遇到错误时终止执行
set -e

# 加载 .env 文件中的环境变量
source .env

# 重启 SSH 服务
echo "$SSH_PASSWORD" | sudo -S service ssh restart

# 启动 Hadoop
/usr/local/hadoop/sbin/start-all.sh

# 启动 Spark
/usr/local/spark/sbin/start-all.sh

echo "All services have been started successfully."