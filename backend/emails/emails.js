import chalk from 'chalk';
import {
  VERIFICATION_EMAIL_TEMPLATE,
  PASSWORD_RESET_SUCCESS_TEMPLATE,
  PASSWORD_RESET_REQUEST_TEMPLATE,
} from './emailTemplates.js';
import transporter from '../config/transporter.js';
import ENV_VARS from '../config/envVars.js';

const sendVerificationEmail = async (email, verificationToken) => {
  try {
    const info = await transporter.sendMail({
      from: `Movie Recommendation < ${ENV_VARS.EMAIL_USER} >`,
      to: email,
      subject: 'Verify your email',
      html: VERIFICATION_EMAIL_TEMPLATE.replace('{verificationCode}',
        verificationToken),
      category: 'Verification',
    })

    console.log(chalk.green.bold(`Verification email sent: ${info.messageId}`));
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in sendVerificationEmail - emails: ${error.message}`));
    throw new Error(
      `Error in sendVerificationEmail - emails: ${error.message}`);
  }
};

// const sendWelcomeEmail = async (email, username) => {
//   try {
//     await mailtrapClient.testing.send({
//       from: sender,
//       to: recipient,
//       template_uuid: '68f882ce-351f-4d6a-87e3-6a334986b526',
//       template_variables: {
//         'company_info_name': 'Movie Recommendation',
//         'name': username,
//       },
//     });
//
//     console.log(chalk.green.bold(`Welcome email sent successfully`));
//   } catch (error) {
//     console.log(
//       chalk.red.bold(`Error in sendWelcomeEmail - emails: ${error.message}`));
//     throw new Error(`Error in sendWelcomeEmail - emails: ${error.message}`);
//   }
// };

const sendResetPasswordEmail = async (email, resetPasswordURL) => {
  try {
    const info = await transporter.sendMail({
      from: `Movie Recommendation < ${ENV_VARS.EMAIL_USER} >`,
      to: email,
      subject: 'Reset your password',
      html: PASSWORD_RESET_REQUEST_TEMPLATE.replace('{resetURL}',
        resetPasswordURL),
      category: 'Reset Password',
    })

    console.log(chalk.green.bold(`Reset password email sent: ${info.messageId}`));
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in sendResetPasswordEmail - emails: ${error.message}`));
    throw new Error(
      `Error in sendResetPasswordEmail - emails: ${error.message}`);
  }
};

const sendResetSuccessEmail = async (email) => {
  try {
    const info = await transporter.sendMail({
      from: `Movie Recommendation < ${ENV_VARS.EMAIL_USER} >`,
      to: email,
      subject: 'Password Reset Successful',
      html: PASSWORD_RESET_SUCCESS_TEMPLATE,
      category: 'Reset Password',
    })

    console.log(chalk.green.bold(`Reset success email sent: ${info.messageId}`));
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in sendResetSuccessEmail - emails: ${error.message}`));
    throw new Error(
      `Error in sendResetSuccessEmail - emails: ${error.message}`);
  }
};

export {
  sendVerificationEmail,
  sendResetPasswordEmail,
  sendResetSuccessEmail,
};