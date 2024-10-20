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
};

export default ENV_VARS;