import { Button } from 'antd';
import useAuthStore from '../store/authStore.js';

const HomePage = () => {
  const { logout } = useAuthStore();

  return (
    <Button onClick={logout}>Home Page</Button>
  );
};

export default HomePage;