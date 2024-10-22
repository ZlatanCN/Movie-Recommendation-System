import chalk from 'chalk';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';
import User from '../models/userModel.js';
import ENV_VARS from '../config/envVars.js';
import generateVerificationToken from '../utils/generateVerificationToken.js';
import generateTokenAndSetCookie from '../utils/generateTokenAndSetCookie.js';
import {
  sendResetPasswordEmail,
  sendVerificationEmail,
  sendResetSuccessEmail,
} from '../emails/emails.js';

const signup = async (req, res) => {
  const { username, password, email } = req.body;

  try {
    const isAllFieldsProvided = username && password && email;
    const isUserNameExist = await User.findOne({ username });
    const isEmailExist = await User.findOne({ email });

    if (!isAllFieldsProvided) {
      throw new Error('Please provide all the required fields!');
    } else if (isUserNameExist) {
      throw new Error('Username already exists!');
    } else if (isEmailExist) {
      throw new Error('Email already exists!');
    } else {
      const hashedPassword = await bcrypt.hash(password, 10);
      const verificationToken = generateVerificationToken();
      const newUser = new User({
        username,
        email,
        password: hashedPassword,
        verificationToken,
        verificationTokenExpiredAt: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
      });

      await newUser.save();

      generateTokenAndSetCookie(res, newUser._id);
      await sendVerificationEmail(email, verificationToken);

      res.status(201).json({
        isSuccessful: true,
        message: 'User created successfully!',
        user: newUser,
      });
    }
  } catch (error) {
    console.log(
      chalk.red.bold(`Error in signup - authController: ${error.message}`),
    );
    res.status(400).json({ isSuccessful: false, message: error.message });
  }
};

const verifyEmail = async (req, res) => {
  const { code } = req.body;

  try {
    const user = await User.findOne({
      verificationToken: code,
      verificationTokenExpiredAt: { $gt: Date.now() },
    });

    if (!user) {
      throw new Error('Invalid or expired verification code!');
    } else {
      user.isVerified = true;
      user.verificationToken = undefined;
      user.verificationTokenExpiredAt = undefined;

      await user.save();
      // await sendWelcomeEmail(user.email, user.username);

      console.log(chalk.green.bold('Email verified successfully!'));
      res.status(200).
        json({
          isSuccessful: true,
          message: 'Email verified successfully!',
          user: user,
        });
    }
  } catch (error) {
    console.log(
      chalk.red.bold(`Error in verifyEmail - authController: ${error.message}`),
    );
    res.status(400).json({ isSuccessful: false, message: error.message });
  }
};

const login = async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await User.findOne({ email });
    // console.log(user);
    const isPasswordMatch = user &&
      (await bcrypt.compare(password, user.password));

    if (!user) {
      throw new Error('User not found!');
    } else if (!isPasswordMatch) {
      throw new Error('Invalid password!');
    } else {
      generateTokenAndSetCookie(res, user._id);

      user.lastLogin = Date.now();
      await user.save();

      res.status(200).json({
        isSuccessful: true,
        message: 'User logged in successfully!',
        user: user,
      });
    }
  } catch (error) {
    console.log(
      chalk.red.bold(`Error in login - authController: ${error.message}`),
    );
    res.status(400).json({ isSuccessful: false, message: error.message });
  }
};

const logout = async (req, res) => {
  res.clearCookie('authToken');
  res.status(200).
    json({ isSuccessful: true, message: 'User logged out successfully!' });
};

const forgotPassword = async (req, res) => {
  const { email } = req.body;

  try {
    const user = await User.findOne({ email });

    if (!user) {
      throw new Error('User not found!');
    } else {
      const resetPasswordToken = crypto.randomBytes(20).toString('hex');
      const resetPasswordExpiredAt = Date.now() + 60 * 60 * 1000; // 1 hour

      user.resetPasswordToken = resetPasswordToken;
      user.resetPasswordExpiredAt = resetPasswordExpiredAt;

      await user.save();

      await sendResetPasswordEmail(
        email,
        `${ENV_VARS.CLIENT_URL}/reset-password/${resetPasswordToken}`,
      );

      res.status(200).json({
        isSuccessful: true,
        message: 'Reset password email sent successfully!',
      });
    }
  } catch (error) {
    console.log(
      chalk.red.bold(
        `Error in forgotPassword - authController: ${error.message}`),
    );
    res.status(400).json({ isSuccessful: false, message: error.message });
  }
};

const resetPassword = async (req, res) => {
  const { password, confirmPassword } = req.body;
  const { token } = req.params;

  try {
    const user = await User.findOne({
      resetPasswordToken: token,
      resetPasswordExpiredAt: { $gt: Date.now() },
    });

    if (!user) {
      throw new Error('Invalid or expired reset password token!');
    } else if (password !== confirmPassword) {
      throw new Error('Passwords do not match!');
    } else {
      user.password = await bcrypt.hash(password, 10);
      user.resetPasswordToken = undefined;
      user.resetPasswordExpiredAt = undefined;

      await user.save();
      await sendResetSuccessEmail(user.email);

      res.status(200).json({
        isSuccessful: true,
        message: 'Password reset successfully!',
      });
    }
  } catch (error) {
    console.log(
      chalk.red.bold(
        `Error in resetPassword - authController: ${error.message}`),
    );
    res.status(400).json({ isSuccessful: false, message: error.message });
  }
};

const checkAuth = async (req, res) => {
  try {
    const user = await User.findById(req.userId);
    // console.log('checkAuth', user);

    if (!user) {
      throw new Error('User not found!');
    } else {
      res.status(200).json({
        isSuccessful: true,
        message: 'User authenticated successfully!',
        user: user,
      });
    }
  } catch (error) {
    console.log(
      chalk.red.bold(`Error in checkAuth - authController: ${error.message}`),
    );
    res.status(400).json({ isSuccessful: false, message: error.message });
  }
};

export {
  signup,
  login,
  logout,
  verifyEmail,
  forgotPassword,
  resetPassword,
  checkAuth,
};