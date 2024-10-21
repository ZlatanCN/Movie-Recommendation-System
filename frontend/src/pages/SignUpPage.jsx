import { Link } from 'react-router-dom';
import SignUpForm from '../components/SignUpForm.jsx';

const SignUpPage = () => {
  const handleSignUp = (e) => {
    e.preventDefault();
  };

  return (
    <div className={'h-screen w-full hero-bg bg-center'}>
      {/* Header */}
      <header
        className={'max-w-6xl mx-auto flex items-center justify-between p-4'}>
        <Link to={'/'}>
          <img
            src={'/netflix-logo.png'}
            alt={'logo'}
            className={'w-52'}
          />
        </Link>
      </header>

      {/*Sign Up Form*/}
      <section className={'flex justify-center items-center mt-20 mx-3 '}>
        <SignUpForm handleSignUp={handleSignUp}/>
      </section>
    </div>
  );
};

export default SignUpPage;