import { Link, useNavigate } from 'react-router-dom';
import SignUpForm from '../components/SignUpForm.jsx';
import NavBar from '../components/NavBar.jsx';
import useAuthStore from '../store/authStore.js';

const SignUpPage = () => {
  const { signup, error, isLoading } = useAuthStore();
  const navigate = useNavigate();

  const handleSignUp = async (e) => {
    e.preventDefault();

    const username = e.target[0].value
    const email = e.target[1].value
    const password = e.target[2].value

    try {
      await signup(username, email, password);
      navigate('/verify-email');
    } catch (error) {
      console.log(`Error in handleSignUp - SignUpPage: ${error.message}`);
    }
  };

  return (
    <div className={'h-screen w-full hero-bg bg-center'}>
      {/* NavBar */}
      <NavBar type={'auth'}/>

      {/*Sign Up Form*/}
      <section className={'flex justify-center items-center mt-20 mx-3 '}>
        <SignUpForm handleSignUp={handleSignUp} isLoading={isLoading} error={error}/>
      </section>
    </div>
  );
};

export default SignUpPage;