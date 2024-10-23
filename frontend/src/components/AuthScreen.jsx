import PropTypes from 'prop-types';
import NavBar from './NavBar.jsx';
import { MailOutlined, RightOutlined } from '@ant-design/icons';
import { ConfigProvider, Input } from 'antd';
import { useState } from 'react';
import { heroInputTheme } from '../theme/inputTheme.js';
import { motion } from 'framer-motion';

const AuthScreen = (props) => {
  const [email, setEmail] = useState('');

  const handleSubmit = async (e) => {

  };

  return (
    <div className={'hero-bg relative'}>
      {/* Navigation Bar */}
      <NavBar type={'auth'}/>

      {/*Hero Section*/}
      <section
        className={'flex flex-col items-center justify-center text-center py-40 text-white max-w-6xl mx-auto gap-4'}>
        <h1 className={'text-4xl md:text-6xl font-bold'}>
          Unlimited movies and more.
        </h1>
        <p className={'text-lg'}>
          Watch anywhere. Cancel anytime.
        </p>
        <p>
          Ready to watch? Enter your email to create or restart your membership.
        </p>
        <form
          onSubmit={handleSubmit}
          className={'flex flex-col md:flex-row gap-4 w-1/2'}
        >
          <ConfigProvider theme={heroInputTheme}>
            <Input
              prefix={<MailOutlined className={'text-red-600 pr-1'}/>}
              placeholder={'Email Address'}
              type={'email'}
              value={email}
              size={'large'}
              required
              onChange={(e) => setEmail(e.target.value)}
              className={'bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 text-white p-2 flex-1'}
            />
          </ConfigProvider>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type={'button'}
            className={'bg-red-700 text-sm lg:text-md px-2 lg:px-6 py-1 md:py-2 rounded-lg flex justify-center items-center font-bold gap-1'}
          >
            Get Started
            <RightOutlined/>
          </motion.button>
        </form>
      </section>

      {/*Separator*/}
      <div className={'h-2 bg-[#232323] w-full aria-hidden:true'}/>

      {/*1st Section*/}
      <section className={'py-10 bg-black text-white'}>
        <main
          className={'flex max-w-6xl mx-auto items-center justify-center md:flex-row flex-col px-4 md:px-2'}>
          {/*Left*/}
          <article className={'flex-1 text-center md:text-left'}>
            <h2 className={'text-4xl md:text-5xl font-extrabold mb-4'}>
              Enjoy on your TV
            </h2>
            <p className={'text-lg md:text-xl'}>
              Watch on Smart TVs, Playstation, Xbox, Chromecast, Apple TV,
              Blu-ray players, and more.
            </p>
          </article>

          {/*Right*/}
          <div className={'flex-1 relative'}>
            <img src={'/tv.png'} alt={'tv'} className={'mt-4 z-20 relative'}/>
            <video
              playsInline={true}
              autoPlay={true}
              muted={true}
              loop={true}
              className={'absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-1/2 z-10'}
            >
              <source src={'/hero-vid.m4v'} type={'video/mp4'}/>
            </video>
          </div>
        </main>
      </section>

      {/*Separator*/}
      <div className={'h-2 bg-[#232323] w-full aria-hidden:true'}/>

      {/*2nd Section*/}
      <section className={'py-10 bg-black text-white'}>
        <main
          className={'flex max-w-6xl mx-auto items-center justify-center md:flex-row flex-col-reverse px-4 md:px-2'}>
          {/*Left*/}
          <div className={'flex-1'}>
            <figure className={'relative'}>
              <img
                src={'/stranger-things-lg.png'}
                alt={'strangerThingsLg'}
                className={'mt-4'}
              />
              <figure
                className={'flex items-center gap-2 absolute bottom-5 left-1/2 -translate-x-1/2 bg-black w-3/4 lg:w-1/2 h-24 border border-slate-500 rounded-md px-2'}>
                <img src={'/stranger-things-sm.png'} alt={'strangerThingsSm'}
                     className={'h-full'}/>
                <figure className={'flex justify-between items-center w-full'}>
                  <div className={'flex flex-col gap-0'}>
                    <span className={'text-md lg:text-lg font-bold'}>Stranger Thing</span>
                    <span
                      className={'text-sm text-blue-500'}>Downloading...</span>
                  </div>
                  <img src={'/download-icon.gif'} alt={'downloadIcon'}
                       className={'h-12'}/>
                </figure>
              </figure>
            </figure>
          </div>

          {/*Right*/}
          <article className={'flex-1 md:text-left text-center'}>
            <h2
              className={'text-4xl md:text-5xl font-extrabold mb-4 text-balance line'}>
              Download your shows to watch offline
            </h2>
            <p className={'text-lg md:text-xl'}>
              Save your favorites easily and always have something to watch.
            </p>
          </article>
        </main>
      </section>

      {/*Separator*/}
      <div className={'h-2 bg-[#232323] w-full aria-hidden:true'}/>

      {/*3rd Section*/}
      <section className={'py-10 bg-black text-white'}>
        <main
          className={'flex max-w-6xl mx-auto items-center justify-center md:flex-row flex-col px-4 md:px-2'}>
          {/*Left*/}
          <article className={'flex-1 text-center md:text-left'}>
            <h2 className={'text-4xl md:text-5xl font-extrabold mb-4'}>
              Watch everywhere
            </h2>
            <p className={'text-lg md:text-xl'}>
              Stream unlimited movies and TV shows on your phone, tablet,
              laptop, and TV.
            </p>
          </article>

          {/*Right*/}
          <div className={'flex-1 relative overflow-hidden'}>
            <img src={'/device-pile.png'} alt={'devicePile'}
                 className={'mt-4 z-20 relative'}/>
            <video
              playsInline={true}
              autoPlay={true}
              muted={true}
              loop={true}
              className={'absolute top-2 left-1/2 -translate-x-1/2 h-4/6 z-10 max-w-[63%]'}
            >
              <source src={'/video-devices.m4v'} type={'video/mp4'}/>
            </video>
          </div>
        </main>
      </section>

      {/*Separator*/}
      <div className={'h-2 bg-[#232323] w-full aria-hidden:true'}/>

      {/*4th Section*/}
      <section className={'py-10 bg-black text-white'}>
        <main
          className={'flex max-w-6xl mx-auto items-center justify-center flex-col-reverse md:flex-row px-4 md:px-2'}>
          {/*Left*/}
          <figure className={'flex-1 relative'}>
            <img src={'/kids.png'} alt={'kids'} className={'mt-4'}/>
          </figure>

          {/*Right*/}
          <article className={'flex-1 text-center md:text-left'}>
            <h2 className={'text-4xl md:text-5xl font-extrabold mb-4'}>
              Create profiles for kids
            </h2>
            <p className={'text-lg md:text-xl'}>
              Send kids on adventures with their favorite characters in a space
              made just for them—free with your membership.
            </p>
          </article>
        </main>
      </section>
    </div>
  );
};

AuthScreen.propTypes = {};

export default AuthScreen;
