import React from 'react';
import { loadingSpinTheme } from '../theme/spinTheme.js';
import { ConfigProvider, Spin } from 'antd';

const LoadingSpin = () => {
  return (
    <ConfigProvider theme={loadingSpinTheme}>
      <Spin
        size={'large'}
        className={'h-screen w-full bg-black bg-center flex justify-center items-center'}
      />
    </ConfigProvider>
  );
};

export default LoadingSpin;
