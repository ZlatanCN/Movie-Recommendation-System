import jwt from 'jsonwebtoken';
import ENV_VARS from '../config/envVars.js';

const generateTokenAndSetCookie = (res, userId) => {
  const token = jwt.sign({ userId }, ENV_VARS.JWT_SECRET, {
    expiresIn: '30d',
  });

  res.cookie('authToken', token, {
    httpOnly: true, // Prevents XSS attacks
    secure: ENV_VARS.NODE_ENV === 'production', // Cookie only sent in HTTPS
    sameSite: 'strict', // CSRF protection
    maxAge: 30 * 24 * 60 * 60 * 1000, // 30 days
  })

  return token;
};

export default generateTokenAndSetCookie;