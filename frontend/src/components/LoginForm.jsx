import PropTypes from 'prop-types';
import { motion } from 'framer-motion';
import { Input } from 'antd';
import {
  EyeInvisibleOutlined,
  EyeOutlined, LoadingOutlined,
  LockOutlined,
  MailOutlined,
} from '@ant-design/icons';
import { useState } from 'react';
import { Link } from 'react-router-dom';

const LoginForm = (props) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={'max-w-md w-full bg-gray-800 bg-opacity-50 backdrop-filter backdrop-blur-xl rounded-2xl shadow-xl overflow-hidden'}
    >
      <section className={'p-8'}>
        <h2
          className={'text-3xl font-bold mb-6 text-center bg-gradient-to-r from-red-600 to-red-700 text-transparent bg-clip-text'}>
          Welcome Back
        </h2>
        <form onSubmit={props.handleLogin} className={'flex flex-col gap-6'}>
          {/* Form Inputs */}
          <Input
            prefix={<MailOutlined className={'text-red-600 pr-1'}/>}
            placeholder={'Email Address'}
            type={'email'}
            value={email}
            size={'large'}
            required
            onChange={(e) => setEmail(e.target.value)}
            className={'bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 text-white'}
          />
          <Input.Password
            prefix={<LockOutlined className={'text-red-600 pr-1'}/>}
            placeholder={'Password'}
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

          {/* Forgot Password */}
          <div className={'flex items-center'}>
            <Link to={'/forgot-password'}
                  className={'text-sm text-red-500 hover:underline'}>
              Forgot Password ?
            </Link>
          </div>

          {/* Submit Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type={'submit'}
            className={'w-full py-3 px-4 bg-gradient-to-r from-red-600 to-red-700 text-white font-bold rounded-lg shadow-lg hover:from-red-700 hover:to-red-800 focus:outline-none focus:ring-1 focus:ring-red-600 focus:ring-offset-1 focus:ring-offset-gray-900'}
          >
            {props.isLoading ? <LoadingOutlined /> : 'Login'}
          </motion.button>
        </form>
      </section>

      {/*Don't have an account ?*/}
      <footer
        className={'px-8 py-4 bg-gray-900 bg-opacity-50 flex justify-center'}>
        <p className={'text-sm text-gray-400'}>
          Don't have an account ?{' '}
          <Link to={'/signup'} className={'text-red-500 hover:underline'}>
            Sign up Now !
          </Link>
        </p>
      </footer>
    </motion.div>
  );
};

LoginForm.PropTypes = {
  handleLogin: PropTypes.func.isRequired,
  isLoading: PropTypes.bool.isRequired,
};

export default LoginForm;