import Header from '../components/Header.jsx';
import LoginForm from '../components/LoginForm.jsx';
import useAuthStore from '../store/authStore.js';
import { message } from 'antd';

const LoginPage = () => {
  const [messageApi, contextHolder] = message.useMessage();
  const { login, error, isLoading, user } = useAuthStore();

  const handleLogin = async (e) => {
    e.preventDefault();

    const email = e.target[0].value
    const password = e.target[1].value

    await login(email, password).then(() => {
      console.log('User logged in successfully!');
    }).catch((error) => {
      // console.log(error.response.data.message);
      messageApi.error({
        content: error.response.data.message || 'Error in logging in!',
        className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
      });
    });
  };

  return (
    <div className={'h-screen w-full hero-bg bg-center'}>
      {contextHolder}

      {/* Header */}
      <Header type={'auth'}/>

      {/* Login Form */}
      <section className={'flex justify-center items-center mt-20 mx-3 '}>
        <LoginForm handleLogin={handleLogin} isLoading={isLoading}/>
      </section>
    </div>
  );
};

export default LoginPage;