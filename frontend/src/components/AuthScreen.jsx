import PropTypes from 'prop-types';
import NavBar from './NavBar.jsx';
import { MailOutlined, RightOutlined } from '@ant-design/icons';
import { ConfigProvider, Input } from 'antd';
import { useState } from 'react';
import { heroInputTheme } from '../theme/inputTheme.js';
import { motion } from 'framer-motion';
import AuthScreenSection from './AuthScreenSection.jsx';
import { useNavigate } from 'react-router-dom';

const AuthScreen = (props) => {
  const [email, setEmail] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    navigate(`/signup?email=${email}`);
  };

  return (
    <div className={'hero-bg relative'}>
      {/* Navigation Bar */}
      <NavBar type={'auth'}/>

      {/*Hero Section*/}
      <section
        className={'flex flex-col items-center justify-center text-center py-40 text-white max-w-6xl mx-auto gap-4'}>
        <h1 className={'text-4xl md:text-6xl font-bold'}>
          Unlimited movies and more.
        </h1>
        <p className={'text-lg'}>
          Watch anywhere. Cancel anytime.
        </p>
        <p>
          Ready to watch? Enter your email to create or restart your membership.
        </p>
        <form
          onSubmit={handleSubmit}
          className={'flex flex-col md:flex-row gap-4 w-1/2'}
        >
          <ConfigProvider theme={heroInputTheme}>
            <Input
              prefix={<MailOutlined className={'text-red-600 pr-1'}/>}
              placeholder={'Email Address'}
              type={'email'}
              value={email}
              size={'large'}
              required
              onChange={(e) => setEmail(e.target.value)}
              className={'bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 text-white p-2 flex-1'}
            />
          </ConfigProvider>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type={'submit'}
            className={'bg-red-700 text-sm lg:text-md px-2 lg:px-6 py-1 md:py-2 rounded-lg flex justify-center items-center font-bold gap-1'}
          >
            Get Started
            <RightOutlined/>
          </motion.button>
        </form>
      </section>

      <AuthScreenSection/>
    </div>
  );
};

AuthScreen.propTypes = {};

export default AuthScreen;
