import jwt from 'jsonwebtoken';
import chalk from 'chalk';
import ENV_VARS from '../config/envVars.js';
import User from '../models/userModel.js';

const verifyToken = async (req, res, next) => {
  const token = req.cookies.authToken;

  try {
    if (!token) throw new Error('No token found!');

    const decoded = jwt.verify(token, ENV_VARS.JWT_SECRET);
    if (!decoded) throw new Error('Invalid token!');

    req.userId = decoded.userId;
    req.user = await User.findById(req.userId);
    next();
  } catch (error) {
    console.log(chalk.red.bold(`Error in verifyToken: ${error.message}`));
    res.status(401).json({ isSuccessful: false, message: 'Unauthorized' });
  }
};

export default verifyToken;