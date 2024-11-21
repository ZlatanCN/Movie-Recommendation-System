import axios from 'axios';
import { useEffect, useRef, useState } from 'react';

const useSimilarMovies = (movieId) => {
  const [similarMovies, setSimilarMovies] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const sliderRef = useRef(null);

  const getSimilarMovies = async () => {
    // try {
    //   const response = await axios.get(`/api/movie/${id}/similar`);
    //   setSimilarContent(response.data.similar);
    // } catch (error) {
    //   if (error.message.includes('404')) {
    //     setSimilarContent([]);
    //   }
    //   console.log(`Error in getSimilarMovies - WatchPage: ${error.message}`);
    // }

    try {
      setIsLoading(true);
      const response = await axios.get(`/api/recommendation/content/${movieId}`);
      setSimilarMovies(response.data.content);
    } catch (error) {
      setSimilarMovies([]);
      console.log(`Error in getSimilarMovies - WatchPage: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    getSimilarMovies();
    sliderRef.current &&
    sliderRef.current.scrollTo({ left: 0, behavior: 'smooth' });
  }, [movieId]);

  return { similarMovies, sliderRef, isLoading };
};

export default useSimilarMovies;