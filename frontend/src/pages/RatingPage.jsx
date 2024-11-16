import { AnimatePresence, motion } from 'framer-motion';
import useRating from '../hooks/useRating.jsx';
import NavBar from '../components/NavBar.jsx';
import axios from 'axios';
import MovieRating from '../components/MovieRating.jsx';
import { message } from 'antd';
import { useState } from 'react';

const RatingPage = () => {
  const { ratedmovies, setRatedmovies } = useRating();
  const [messageApi, contextHolder] = message.useMessage();
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 4;
  const totalPages = Math.ceil(ratedmovies.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentMovies = ratedmovies.slice(startIndex, startIndex + itemsPerPage);

  const handleDelete = async (movie) => {
    try {
      const response = await axios.delete(`/api/rating/${movie.id}`);
      if (response.data.isSuccessful) {
        messageApi.success({
          content: 'Rating deleted successfully!',
          className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
        });
        setRatedmovies(ratedmovies.filter((ratedmovie) => ratedmovie.id !== movie.id));
      } else {
        throw new Error('Failed to delete rating!');
      }
    } catch (error) {
      console.error(error);
      messageApi.error({
        content: error.response.data.message || 'Error in deleting rating!',
        className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
      });
    }
  };

  const updateRating = async (id, rating) => {
    try {
      const response = await axios.post(`/api/rating/${id}`, { rating });
      if (response.data.isSuccessful) {
        messageApi.success({
          content: 'Rating updated successfully!',
          className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
        });
        setRatedmovies(ratedmovies.map((movie) => movie.id === id ? { ...movie, rating } : movie));
      } else {
        throw new Error('Failed to update rating!');
      }
    } catch (error) {
      console.error(error);
      messageApi.error({
        content: error.response.data.message || 'Error in updating rating!',
        className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
      });
    }
  }

  return (
    <>
      {contextHolder}

      <div className={'bg-black text-white min-h-screen relative'}>
        <NavBar/>
        <section className={'max-w-6xl mx-auto px-4 py-8'}>
          <h1 className={'text-3xl font-bold mb-8'}>
            Your Ratings
          </h1>
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={'grid grid-cols-1 sm:grid-cols-1 md:grid-cols-1 lg:grid-cols-2 gap-4'}
          >
            <AnimatePresence>
              {currentMovies.map((movie) => (
                <motion.div
                  key={movie.id}
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -50 }}
                  transition={{ duration: 0.8, ease: 'easeInOut' }}
                >
                  <MovieRating handleDelete={handleDelete} movie={movie} updateRating={updateRating}/>
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        </section>

        {/* Absolute Pagination */}
        <motion.section
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className={'absolute bottom-0 left-0 right-0 bg-black py-4 flex justify-between px-4'}
        >
          <button
            onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className={`bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-red-700 ${currentPage === 1 ? 'cursor-not-allowed opacity-50' : 'opacity-70'}`}
          >
            Prev
          </button>
          <button
            onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            className={`bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-red-700 ${currentPage === totalPages ? 'cursor-not-allowed opacity-50' : 'opacity-70'}`}
          >
            Next
          </button>
        </motion.section>
      </div>
    </>
  );
};

export default RatingPage;