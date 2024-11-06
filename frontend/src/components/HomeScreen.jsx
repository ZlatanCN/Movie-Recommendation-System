import NavBar from './NavBar.jsx';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  InfoCircleOutlined,
  PlayCircleFilled,
} from '@ant-design/icons';
import useGetTrendingMovie from '../hooks/useGetTrendingMovie.jsx';
import { MOVIE_CATEGORIES, ORIGINAL_IMG_BASE_URL } from '../utils/constants.js';
import MovieSlider from './MovieSlider.jsx';
import LoadingSpin from './LoadingSpin.jsx';

const HomeScreen = () => {
  const { trendingMovie } = useGetTrendingMovie();

  if (!trendingMovie) {
    return (
      <LoadingSpin/>
    )
  }

  return (
    <>
      {/*Hero Section*/}
      <section
        className={'relative h-screen text-white'}
      >
        {/* Navigation Bar */}
        <NavBar/>

        {/*Background Image*/}
        <img
          src={ORIGINAL_IMG_BASE_URL + trendingMovie?.backdrop_path}
          alt={'Backdrop'}
          className={'absolute top-0 left-0 w-full h-full object-cover -z-50'}
        />
        <div
          className={'absolute top-0 left-0 w-full h-full bg-black/50 -z-50 aria-hidden:true'}/>

        {/*Movie Outline*/}
        <section
          className={'absolute top-0 left-0 w-full h-full flex flex-col justify-center px-8 md:px-16 lg:px-32'}>
          {/*Shadow Overlay*/}
          <div
            className={'bg-gradient-to-b from-black via-transparent to-transparent absolute w-full h-full top-0 left-0 -z-10'}/>

          {/*Movie Info*/}
          <main className={'max-w-2xl'}>
            <h1 className={'mt-4 text-6xl font-extrabold text-balance'}>
              {trendingMovie?.title || trendingMovie?.name}
            </h1>
            <p className={'mt-2 text-lg'}>
              {trendingMovie?.release_date?.split('-')[0]}
              {' | '}
              {trendingMovie?.adult ? '18+' : 'PG-13'}
            </p>
            <p className={'mt-4 text-lg'}>
              {trendingMovie?.overview}
            </p>
          </main>

          {/*Movie Play Button*/}
          <button className={'flex mt-8'}>
            <Link
              to={`/watch/${trendingMovie?.id}`}
              className={'bg-white hover:bg-white/80 text-black font-bold py-2 px-4 rounded mr-4 flex items-center'}
            >
              <PlayCircleFilled className={'pr-1.5'}/>
              Play
            </Link>
            <Link
              to={'/'}
              className={'bg-gray-500/70 hover:bg-gray-500 text-white py-2 px-4 flex items-center rounded font-bold'}
            >
              <InfoCircleOutlined className={'pr-1.5'}/>
              More Info
            </Link>
          </button>
        </section>
      </section>

      {/*Categories Section*/}
      <section className={'flex flex-col gap-10 bg-black py-10'}>
        {MOVIE_CATEGORIES.map((category) => (
          <MovieSlider key={category} category={category}/>
        ))}
      </section>
    </>
  );
};

export default HomeScreen;
