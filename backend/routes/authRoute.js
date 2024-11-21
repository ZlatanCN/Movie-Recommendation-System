import express from 'express';
import {
  checkAuth,
  forgotPassword,
  login,
  logout,
  resetPassword,
  signup,
  verifyEmail,
} from '../controllers/authController.js';
import verifyToken from '../middleware/verifyToken.js';

const router = express.Router();

// @desc    Check if user is authenticated
// @route   GET /api/auth/check-auth
router.get('/check-auth', verifyToken, checkAuth);

// @desc    Register a new user
// @route   POST /api/auth/signup
router.post('/signup', signup);

// @desc    Verify user's email
// @route   POST /api/auth/verify-email
router.post('/verify-email', verifyEmail);

// @desc    Login an existing user
// @route   POST /api/auth/login
router.post('/login', login);

// @desc    Logout a user
// @route   POST /api/auth/logout
router.post('/logout', logout);

// @desc    Forgot password
// @route   POST /api/auth/forgot-password
router.post('/forgot-password', forgotPassword);

// @desc    Reset password
// @route   POST /api/auth/reset-password/:token
router.post('/reset-password/:token', resetPassword);

export default router;