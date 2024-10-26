import PropTypes from 'prop-types';
import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { Link } from 'react-router-dom';
import {
  ORIGINAL_IMG_BASE_URL,
  SMALL_IMG_BASE_URL,
} from '../utils/constants.js';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';

const MovieSlider = (props) => {
  const [movies, setMovies] = useState([]);
  const [showArrow, setShowArrow] = useState(false);
  const sliderRef = useRef(null);

  const formatCategory = props.category.replace(/_/g, ' ').
    split(' ').
    map(word => word.charAt(0).toUpperCase() + word.slice(1)).
    join(' ');

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
    const getMovies = async () => {
      const response = await axios.get(`/api/movie/${props.category}`);
      setMovies(response.data.content);
    };

    getMovies();
  }, []);

  return (
    <div
      onMouseEnter={() => setShowArrow(true)}
      onMouseLeave={() => setShowArrow(false)}
      className={'text-white bg-black relative px-5 md:px-20'}
    >
      <h2 className={'font-bold mb-4 text-2xl'}>
        {formatCategory} Movies
      </h2>

      <div
        className={'flex space-x-4 overflow-x-scroll scrollbar-hide'}
        ref={sliderRef}
      >
        {movies.map((movie) => (
          <Link
            key={movie.id}
            to={`/watch/${movie.id}`}
            className={'min-w-[250px] relative group'}
          >
            <div
              className={'rounded-lg overflow-hidden transition-transform duration-300 ease-in-out group-hover:scale-110'}>
              <img
                src={SMALL_IMG_BASE_URL + movie.backdrop_path}
                alt={'smallBackdrop'}
                className={''}
              />
              <p className={'mt-2 text-center font-semibold'}>
                {movie.title}
              </p>
            </div>
          </Link>
        ))}
      </div>

      {showArrow && (
        <>
          <motion.button
            onClick={scrollX.bind(this, 'left')}
            whileTap={{ opacity: 0.5 }}
            className={'absolute top-1/2 -translate-y-1/2 left-5 md:left-15 flex items-center justify-center size-12 rounded-full bg-gray-800 bg-opacity-50 hover:bg-opacity-60 text-white z-10'}>
            <LeftOutlined/>
          </motion.button>
          <motion.button
            onClick={scrollX.bind(this, 'right')}
            whileTap={{ opacity: 0.5 }}
            className={'absolute top-1/2 -translate-y-1/2 right-5 md:right-15 flex items-center justify-center size-12 rounded-full bg-gray-800 bg-opacity-50 hover:bg-opacity-60 text-white z-10'}>
            <RightOutlined/>
          </motion.button>
        </>
      )}
    </div>
  );
};

MovieSlider.propTypes = {
  category: PropTypes.string.isRequired,
};

export default MovieSlider;
