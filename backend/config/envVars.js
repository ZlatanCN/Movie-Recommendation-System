import dotenv from 'dotenv';

dotenv.config();

const ENV_VARS = {
  PORT: process.env.PORT || 5000,
  MONGO_URI: process.env.MONGO_URI,
  JWT_SECRET: process.env.JWT_SECRET,
  NODE_ENV: process.env.NODE_ENV,
  MAILTRAP_TOKEN: process.env.MAILTRAP_TOKEN,
  MAILTRAP_INBOX_ID: process.env.MAILTRAP_TEST_INBOX_ID,
  CLIENT_URL: process.env.CLIENT_URL,
  TMDB_ACCESS_TOKEN: process.env.TMDB_ACCESS_TOKEN,
  EMAIL_USER: process.env.EMAIL_USER,
  EMAIL_PASS: process.env.EMAIL_PASS,
  MYSQL_HOST: process.env.MYSQL_HOST,
  MYSQL_PORT: process.env.MYSQL_PORT,
  MYSQL_USER: process.env.MYSQL_USER,
  MYSQL_PASSWORD: process.env.MYSQL_PASSWORD,
  MYSQL_DB: process.env.MYSQL_DB,
};

export default ENV_VARS;