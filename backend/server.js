import express from 'express';
import chalk from 'chalk';
import cookieParser from 'cookie-parser';
import ENV_VARS from './config/envVars.js';
import connectDB from './config/db.js';
import authRoute from './routes/authRoute.js';
import movieRoute from './routes/movieRoute.js';
import searchRoute from './routes/searchRoute.js';
import verifyToken from './middleware/verifyToken.js';

const PORT = ENV_VARS.PORT;

const app = express();

app.use(express.json()); // Allows us to parse JSON in the request body
app.use(cookieParser()); // Allows us to parse cookies in the request headers

app.use('/api/auth', authRoute);
app.use('/api/movie', verifyToken, movieRoute);
app.use('/api/search', verifyToken, searchRoute);

app.listen(PORT, () => {
  console.log(chalk.green.bold(`\nServer started at http://localhost:${PORT}`));
  connectDB();
});