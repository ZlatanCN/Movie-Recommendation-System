import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const NavBar = (props) => {
  return (
    <>
      {props.type === 'auth' ? (
        <header
          className={'max-w-6xl mx-auto flex items-center justify-between p-4 pb-10'}>
          <img src={'/netflix-logo.png'} alt={'netflixLogo'} className={'w-32 md:w-52'}/>
          <Link to={'/login'} >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type={'button'}
              className={'text-white bg-red-700 py-2 px-2 rounded-lg text-sm font-semibold'}
            >
              SIGN IN
            </motion.button>
          </Link>
        </header>
      ) : (
        <header className={'max-w-6xl mx-auto flex items-center justify-between p-4'}>

        </header>
      )}
    </>
  );
};

NavBar.propTypes = {
  type: PropTypes.string.isRequired,
};

export default NavBar;