import { Navigate, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage.jsx';
import SignUpPage from './pages/SignUpPage.jsx';
import LoginPage from './pages/LoginPage.jsx';
import EmailVerificationPage from './pages/EmailVerificationPage.jsx';
import useAuthStore from './store/authStore.js';
import { useEffect } from 'react';
import { ConfigProvider, Spin } from 'antd';
import ForgotPasswordPage from './pages/ForgotPasswordPage.jsx';
import ResetPasswordPage from './pages/ResetPasswordPage.jsx';
import Footer from './components/Footer.jsx';

const App = () => {
  const { isCheckingAuth, checkAuth, isAuthenticated, user } = useAuthStore();

  useEffect(() => {
    const checkAuthentication = async () => {
      await checkAuth();

    };
    checkAuthentication()
  }, [checkAuth]);

  console.log(`isAuthenticated: ${isAuthenticated}, user: ${user}, isCheckingAuth: ${isCheckingAuth}`);
  return (
    <>
      {isCheckingAuth ? (
        <ConfigProvider
          theme={{
            token: {
              colorPrimary: 'rgb(239, 68, 68)',
          }}}
        >
          <Spin
            size={'large'}
            className={'h-screen w-full hero-bg bg-center flex justify-center items-center'}
          />
        </ConfigProvider>
      ) : (
        <Routes>
          <Route path={'/'} element={isAuthenticated ? <HomePage/> : <Navigate to={'/login'}/>}/>
          <Route path={'/signup'} element={isAuthenticated ? <Navigate to={'/'}/> : <SignUpPage/>}/>
          <Route path={'/login'} element={isAuthenticated ? <Navigate to={'/'}/> : <LoginPage/>}/>
          <Route
            path={'/verify-email'}
            element={isAuthenticated && user.isVerified ? <Navigate to={'/'}/> : <EmailVerificationPage/>}
          />
          <Route path={'/forgot-password'} element={isAuthenticated ? <Navigate to={'/'}/> : <ForgotPasswordPage/>}/>
          <Route path={'/reset-password/:token'} element={isAuthenticated ? <Navigate to={'/'}/> : <ResetPasswordPage/>}/>
        </Routes>
      )}

      <Footer/>
    </>
  );
};

export default App;
