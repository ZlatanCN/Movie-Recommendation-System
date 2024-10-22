import { Navigate, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage.jsx';
import SignUpPage from './pages/SignUpPage.jsx';
import LoginPage from './pages/LoginPage.jsx';
import EmailVerificationPage from './pages/EmailVerificationPage.jsx';
import useAuthStore from './store/authStore.js';
import { useEffect } from 'react';

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
    <Routes>
      <Route path={'/'} element={isAuthenticated ? <HomePage/> : <Navigate to={'/login'}/>}/>
      <Route path={'/signup'} element={isAuthenticated ? <Navigate to={'/'}/> : <SignUpPage/>}/>
      <Route path={'/login'} element={isAuthenticated ? <Navigate to={'/'}/> : <LoginPage/>}/>
      <Route path={'/verify-email'} element={<EmailVerificationPage/>}/>
    </Routes>
  );
};

export default App;
