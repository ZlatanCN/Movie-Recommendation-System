import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useState } from 'react';
import {
  LogoutOutlined,
  MenuOutlined, QuestionCircleOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import useAuthStore from '../store/authStore.js';
import { ConfigProvider, Popconfirm } from 'antd';
import { logoutPopConfirmTheme } from '../theme/popConfirmTheme.js';

const NavBar = (props) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const { user, logout } = useAuthStore();

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  switch (props.type) {
    case 'auth':
      return (
        <header
          className={'max-w-6xl mx-auto flex items-center justify-between p-4 pb-10'}>
          <img src={'/netflix-logo.png'} alt={'netflixLogo'}
               className={'w-32 md:w-52'}/>
          <Link to={'/login'}>
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
      );
    default:
      return (
        <header
          className={'max-w-6xl mx-auto flex flex-wrap items-center justify-between p-4 h-20'}>
          <nav className={'flex items-center gap-10 z-50'}>
            <Link to={'/'}>
              <img src={'/netflix-logo.png'} alt={'netflixLogo'}
                   className={'w-32 sm:w-40'}/>
            </Link>

            {/*Desktop Navigation*/}
            <nav className={'hidden sm:flex gap-8 items-center'}>
              <Link to={'/'} className={'hover:underline font-semibold'}>
                Movies
              </Link>
              <Link to={'/history'} className={'hover:underline font-semibold'}>
                History
              </Link>
            </nav>
          </nav>

          <div className={'flex gap-4 items-center z-50'}>
            <motion.div
              whileHover={{ scale: 1.08 }}
              whileTap={{ scale: 0.92 }}
              className={'bg-transparent '}
            >
              <Link to={'/search'}>
                <SearchOutlined/>
              </Link>
            </motion.div>

            <img src={'/avatar2.jpg'} alt={'avatar'}
                 className={'h-8 rounded-full cursor-pointer'}/>

            <motion.div
              whileHover={{ scale: 1.08 }}
              whileTap={{ scale: 0.92 }}
              className={'bg-transparent '}
            >
              <ConfigProvider theme={logoutPopConfirmTheme}>
                <Popconfirm
                  title={'Leave Netflix?'}
                  description={'Are you sure you want to logout?'}
                  onConfirm={logout}
                  okText={<span className={'font-semibold'}>Yes</span>}
                  cancelText={<span className={'font-semibold'}>No</span>}
                  color={'rgb(31 41 55)'}
                  icon={
                    <QuestionCircleOutlined
                      style={{
                        color: 'rgb(220 38 38 / 1)',
                      }}
                    />
                  }
                >
                  <LogoutOutlined/>
                </Popconfirm>
              </ConfigProvider>
            </motion.div>

            <motion.menu
              whileHover={{ scale: 1.08 }}
              whileTap={{ scale: 0.92 }}
              className={'sm:hidden bg-transparent'}
            >
              <MenuOutlined onClick={toggleMobileMenu}/>
            </motion.menu>
          </div>

          {/*Mobile Navigation*/}
          {isMobileMenuOpen && (
            <motion.nav
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.38 }}
              className={'w-full sm:hidden mt-4 z-50 bg-black border rounded border-gray-800'}>
              <Link
                to={'/'}
                onClick={toggleMobileMenu}
                className={'block p-2 hover:underline'}
              >
                Movies
              </Link>
              <Link
                to={'/history'}
                className={'block p-2 hover:underline'}
              >
                Search History
              </Link>
            </motion.nav>
          )}
        </header>
      );
  }
};

NavBar.propTypes = {
  type: PropTypes.string,
};

export default NavBar;