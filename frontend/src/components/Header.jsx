import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';

const Header = (props) => {
  return (
    <>
      {props.type === 'auth' ? (
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
      ) : (
        <header className={'max-w-6xl mx-auto flex items-center justify-between p-4'}>

        </header>
      )}
    </>
  );
};

Header.propTypes = {
  type: PropTypes.string.isRequired,
};

export default Header;