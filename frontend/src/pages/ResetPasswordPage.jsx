import PropTypes from 'prop-types';
import { useState } from 'react';
import Header from '../components/Header.jsx';
import { motion } from 'framer-motion';
import { useNavigate, useParams } from 'react-router-dom';
import {
  EyeInvisibleOutlined,
  EyeOutlined, LoadingOutlined,
  LockOutlined,
} from '@ant-design/icons';
import { Input, message } from 'antd';
import useAuthStore from '../store/authStore.js';

const ResetPasswordPage = (props) => {
  const [messageApi, contextHolder] = message.useMessage();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const { token } = useParams();
  const navigate = useNavigate();
  const { isLoading, resetPassword } = useAuthStore();

  const handleSubmit = async (e) => {
    e.preventDefault();

    await resetPassword(token, password, confirmPassword).then(() => {
      messageApi.success({
        content: 'Password reset successfully!',
        className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
      });
      setTimeout(() => navigate('/login'), 2000);
    }).catch((error) => {
      messageApi.error({
        content: error.response.data.message || 'Error in resetting password!',
        className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
      });
      console.log(error.response.data);
    });
  };

  return (
    <div className={'h-screen w-full hero-bg bg-center'}>
      {contextHolder}

      {/* Header */}
      <Header type={'auth'}/>

      <section className={'flex justify-center items-center mt-20 mx-3 '}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className={'max-w-md w-full bg-gray-800 bg-opacity-50 backdrop-filter backdrop-blur-xl rounded-2xl shadow-xl overflow-hidden'}
        >
          <div className={'p-8'}>
            {/*Reset Password Title*/}
            <h2
              className={'text-3xl font-bold text-center bg-gradient-to-r from-red-600 to-red-700 text-transparent bg-clip-text mb-6'}>
              Reset Password
            </h2>

            {/*Reset Password Form*/}
            <form onSubmit={handleSubmit} className={'flex flex-col gap-6'}>
              <Input.Password
                prefix={<LockOutlined className={'text-red-600 pr-1'}/>}
                placeholder={'New Password'}
                type={'password'}
                value={password}
                size={'large'}
                required
                onChange={(e) => setPassword(e.target.value)}
                visibilityToggle={true}
                iconRender={visible => (
                  visible
                    ? <EyeOutlined style={{ color: 'rgb(220 38 38 / 1)' }}/>
                    : <EyeInvisibleOutlined
                      style={{ color: 'rgb(220 38 38 / 1)' }}/>
                )}
                className={'bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 text-white'}
              />
              <Input.Password
                prefix={<LockOutlined className={'text-red-600 pr-1'}/>}
                placeholder={'Confirm New Password'}
                type={'password'}
                value={confirmPassword}
                size={'large'}
                required
                onChange={(e) => setConfirmPassword(e.target.value)}
                visibilityToggle={true}
                iconRender={visible => (
                  visible
                    ? <EyeOutlined style={{ color: 'rgb(220 38 38 / 1)' }}/>
                    : <EyeInvisibleOutlined
                      style={{ color: 'rgb(220 38 38 / 1)' }}/>
                )}
                className={'bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 text-white'}
              />

              {/* Submit Button */}
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type={'submit'}
                className={'w-full py-3 px-4 bg-gradient-to-r from-red-600 to-red-700 text-white font-bold rounded-lg shadow-lg hover:from-red-700 hover:to-red-800 focus:outline-none focus:ring-1 focus:ring-red-600 focus:ring-offset-1 focus:ring-offset-gray-900'}
              >
                {isLoading ? <LoadingOutlined/> : 'Set New Password'}
              </motion.button>
            </form>
          </div>
        </motion.div>
      </section>
    </div>
  );
};

ResetPasswordPage.propTypes = {};

export default ResetPasswordPage;
