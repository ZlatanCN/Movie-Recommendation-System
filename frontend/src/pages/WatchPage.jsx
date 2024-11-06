import { Link, useParams } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import NavBar from '../components/NavBar.jsx';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';
import ReactPlayer from 'react-player';
import {
  ORIGINAL_IMG_BASE_URL,
  SMALL_IMG_BASE_URL,
} from '../utils/constants.js';
import { motion } from 'framer-motion';
import useWatch from '../hooks/useWatch.jsx';
import formatDate from '../utils/formatDate.js';
import LoadingSpin from '../components/LoadingSpin.jsx';

const WatchPage = () => {
  const { id } = useParams();
  const { trailers, content, similarContent, isLoading } = useWatch(id);
  const [currentTrailerIndex, setCurrentTrailerIndex] = useState(0);
  const sliderRef = useRef(null);

  const handleDirection = (direction) => {
    if (direction === 'left') {
      if (currentTrailerIndex > 0) {
        setCurrentTrailerIndex(currentTrailerIndex - 1);
      }
    } else if (direction === 'right') {
      if (currentTrailerIndex < trailers.length - 1) {
        setCurrentTrailerIndex(currentTrailerIndex + 1);
      }
    }
  };

  const scrollX = (direction) => {
    if (sliderRef.current) {
      const distance = -sliderRef.current.offsetWidth;
      if (direction === 'left') {
        sliderRef.current.scrollBy({ left: distance, behavior: 'smooth' });
      } else if (direction === 'right') {
        sliderRef.current.scrollBy({ left: -distance, behavior: 'smooth' });
      }
    }
  };

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    setCurrentTrailerIndex(0);
    sliderRef.current &&
    sliderRef.current.scrollTo({ left: 0, behavior: 'smooth' });
  }, [id]);

  if (isLoading == null) {
    return (
      <LoadingSpin/>
    );
  }

  return (
    <>
      {content.id === undefined ? (
        <div className={'bg-black h-screen text-white'}>
          <div className={'max-w-6xl mx-auto'}>
            <NavBar/>
            <section className={'text-center mx-auto px-4 py-8 h-full mt-48'}>
              <h2 className={'text-2xl sm:text-5xl font-bold text-balance'}>
                Content not found 😢
              </h2>
            </section>
          </div>
        </div>
      ) : (
        <div className={'bg-black min-h-screen text-white'}>
          <div className={'mx-auto container px-4 py-8 h-full'}>
            {/* NavBar */}
            <NavBar/>

            {/* Movie Details */}
            <section
              className={'flex flex-col md:flex-row items-center justify-between gap-20 max-w-6xl mx-auto'}>
              <div className={'mb-4 md:mb-0'}>
                <h2 className={'text-5xl font-bold text-balance'}>
                  {content?.title || 'Movie Title'}
                </h2>
                <p className={'mt-2 text-lg'}>
                  {formatDate(content?.release_date)} | {' '}
                  {content?.adult ? (
                    <span className={'text-red-600'}>18+</span>
                  ) : (
                    <span className={'text-green-600'}>PG-13</span>
                  )}{' '}
                </p>
                <p className={'mt-4 text-lg'}>
                  {content?.overview}
                </p>
              </div>
              <img
                src={ORIGINAL_IMG_BASE_URL + content?.poster_path}
                alt={'poster'}
                className={'max-h-[600px] rounded-md'}
              />
            </section>

            {/* Trailer Video */}
            <section
              className={'relative aspect-video mb-8 p-2 sm:px-10 md:px-32 mt-16'}>
              {trailers.length > 0 ? (
                <>
                  <motion.div
                    key={currentTrailerIndex}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1, ease: 'easeInOut' }}
                  >
                    <ReactPlayer
                      controls={true}
                      width={'100%'}
                      height={'70vh'}
                      url={`https://www.youtube.com/watch?v=${trailers[currentTrailerIndex].key}`}
                      className={'mx-auto overflow-hidden rounded-lg'}
                    />
                  </motion.div>
                  {/* Trailers Buttons */}
                  {trailers.length > 0 && (
                    <>
                      <motion.button
                        className={`absolute left-0 top-1/2 transform -translate-y-1/2 bg-gray-500/70 text-white py-2 px-4 rounded-xl hover:bg-red-700 ${
                          currentTrailerIndex === 0
                            ? 'cursor-not-allowed opacity-50'
                            : 'opacity-70'}`}
                        whileTap={{ opacity: 0.5 }}
                        disabled={currentTrailerIndex === 0}
                        onClick={handleDirection.bind(this, 'left')}
                      >
                        <LeftOutlined/>
                      </motion.button>
                      <motion.button
                        className={`absolute right-0 top-1/2 transform -translate-y-1/2 bg-gray-500/70 text-white py-2 px-4 rounded-xl hover:bg-red-700 ${
                          currentTrailerIndex === trailers.length - 1
                            ? 'cursor-not-allowed opacity-50'
                            : 'opacity-70'}`}
                        whileTap={{ opacity: 0.5 }}
                        disabled={currentTrailerIndex === trailers.length - 1}
                        onClick={handleDirection.bind(this, 'right')}
                      >
                        <RightOutlined/>
                      </motion.button>
                    </>
                  )}
                </>
              ) : (
                <h2 className={'text-xl text-center mt-5'}>
                  No trailers available for{' '}
                  <span className={'font-bold text-red-600'}>
                    {content?.title || 'this movie'}
                  </span>
                </h2>
              )}
            </section>

            {/* Similar Movies */}
            {similarContent.length > 0 && (
              <section className={'mt-12 max-w-5xl mx-auto relative'}>
                <h3 className={'text-3xl font-bold mb-4'}>
                  Similar Movies
                </h3>
                <div
                  className={'flex overflow-x-scroll scrollbar-hide gap-4 pb-4 group'}
                  ref={sliderRef}>
                  {similarContent.map((content) => {
                    if (!content?.poster_path) return null;
                    return (
                      <Link
                        key={content._id}
                        to={`/watch/${content.id}`}
                        className={'w-52 flex-none'}
                      >
                        <img
                          src={SMALL_IMG_BASE_URL + content?.poster_path}
                          alt={content.title}
                          className={'w-full h-auto rounded-md'}
                        />
                        <h4 className={'mt-2 text-lg font-semibold'}>
                          {content.title}
                        </h4>
                      </Link>
                    );
                  })}

                  <motion.button
                    onClick={scrollX.bind(this, 'left')}
                    whileTap={{ opacity: 0.5 }}
                    className={'absolute top-[45%] -translate-y-1/2 -left-16 flex items-center justify-center size-12 rounded-full bg-gray-800 bg-opacity-50 hover:bg-opacity-60 text-white z-10 hover:bg-red-600'}>
                    <LeftOutlined/>
                  </motion.button>
                  <motion.button
                    onClick={scrollX.bind(this, 'right')}
                    whileTap={{ opacity: 0.5 }}
                    className={'absolute top-[45%] -translate-y-1/2 -right-16 flex items-center justify-center size-12 rounded-full bg-gray-800 bg-opacity-50 hover:bg-opacity-60 text-white z-10 hover:bg-red-600'}>
                    <RightOutlined/>
                  </motion.button>
                </div>
              </section>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default WatchPage;