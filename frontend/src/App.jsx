import { Navigate, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage.jsx';
import SignUpPage from './pages/SignUpPage.jsx';
import LoginPage from './pages/LoginPage.jsx';
import EmailVerificationPage from './pages/EmailVerificationPage.jsx';
import useAuthStore from './store/authStore.js';
import { useEffect } from 'react';
import ForgotPasswordPage from './pages/ForgotPasswordPage.jsx';
import ResetPasswordPage from './pages/ResetPasswordPage.jsx';
import Footer from './components/Footer.jsx';
import LoadingSpin from './components/LoadingSpin.jsx';
import WatchPage from './pages/WatchPage.jsx';
import SearchPage from './pages/SearchPage.jsx';
import HistoryPage from './pages/HistoryPage.jsx';
import NotFoundPage from './pages/NotFoundPage.jsx';
import RatingPage from './pages/RatingPage.jsx';
import RecommendationPage from './pages/RecommendationPage.jsx'

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
        <LoadingSpin/>
      ) : (
        <Routes>
          <Route path={'/'} element={<HomePage/>}/>
          <Route path={'/signup'} element={isAuthenticated ? <Navigate to={'/'}/> : <SignUpPage/>}/>
          <Route path={'/login'} element={isAuthenticated ? <Navigate to={'/'}/> : <LoginPage/>}/>
          <Route path={'/verify-email'} element={isAuthenticated && user.isVerified ? <Navigate to={'/'}/> : <EmailVerificationPage/>}/>
          <Route path={'/forgot-password'} element={isAuthenticated ? <Navigate to={'/'}/> : <ForgotPasswordPage/>}/>
          <Route path={'/reset-password/:token'} element={isAuthenticated ? <Navigate to={'/'}/> : <ResetPasswordPage/>}/>
          <Route path={'/watch/:id'} element={isAuthenticated ? <WatchPage/> : <Navigate to={'/'}/>}/>
          <Route path={'/search'} element={isAuthenticated ? <SearchPage/> : <Navigate to={'/'}/>}/>
          <Route path={'/history'} element={isAuthenticated ? <HistoryPage/> : <Navigate to={'/'}/>}/>
          <Route path={'/rating'} element={isAuthenticated ? <RatingPage/> : <Navigate to={'/'}/>}/>
          <Route path={'/recommendation'} element={isAuthenticated ? <RecommendationPage/> : <Navigate to={'/'}/>}/>
          <Route path={'/*'} element={<NotFoundPage/>}/>
        </Routes>
      )}

      <Footer/>
    </>
  );
};

export default App;
