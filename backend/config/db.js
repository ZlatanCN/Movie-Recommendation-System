import mongoose from 'mongoose';
import chalk from 'chalk';
import ENV_VARS from './envVars.js';

const connectDB = async () => {
  try {
    const connection = await mongoose.connect(ENV_VARS.MONGO_URI)
    console.log(chalk.green.bold(`MongoDB connected: ${connection.connection.host}`));
  } catch (error) {
    process.exit(1); // Exit with failure
    console.log(chalk.red.bold(`Error in connecting to MongoDB: ${error.message}`));
  }
}

export default connectDB;