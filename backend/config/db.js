import mongoose from 'mongoose';
import mysql from 'mysql2/promise';
import chalk from 'chalk';
import ENV_VARS from './envVars.js';

let mysqlPool;

const connectMongoDB = async () => {
  try {
    const connection = await mongoose.connect(ENV_VARS.MONGO_URI)
    console.log(chalk.green.bold(`MongoDB connected: ${connection.connection.host}`));
  } catch (error) {
    process.exit(1); // Exit with failure
    console.log(chalk.red.bold(`Error in connecting to MongoDB: ${error.message}`));
  }
}

const connectMySQL = async () => {
  try {
    mysqlPool = await mysql.createPool({
      host: ENV_VARS.MYSQL_HOST,
      port: ENV_VARS.MYSQL_PORT,
      user: ENV_VARS.MYSQL_USER,
      password: ENV_VARS.MYSQL_PASSWORD,
      database: ENV_VARS.MYSQL_DB,
      waitForConnections: true,
      connectionLimit: 10,
      queueLimit: 0
    });
    const connection = await mysqlPool.getConnection();
    console.log(chalk.green.bold('MySQL connected successfully'));
    connection.release(); // 释放连接回到池中
  } catch (error) {
    console.log(chalk.red.bold(`Error in connecting to MySQL: ${error.message}`));
    process.exit(1); // Exit with failure
  }
}

const getMySQLPool = () => {
  if (!mysqlPool) {
    throw new Error('MySQL pool has not been initialized. Call connectMySQL first.');
  }
  return mysqlPool;
};

export { connectMongoDB, connectMySQL, getMySQLPool };