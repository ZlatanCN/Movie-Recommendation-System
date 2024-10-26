import useAuthStore from '../store/authStore.js';
import HomeScreen from '../components/HomeScreen.jsx';
import AuthScreen from '../components/AuthScreen.jsx';

const HomePage = () => {
  const { user } = useAuthStore();



  return (
    <>
      {user ? (
        <HomeScreen/>
      ) : (
        <AuthScreen/>
      )}
    </>
  );
};

export default HomePage;