import { useEffect, useState } from 'react';
import axios from 'axios';

const useGetTrendingMovie = () => {
  const [trendingMovie, setTrendingMovie] = useState(null);

  useEffect(() => {
    const getTrendingMovie = async () => {
      const response = await axios.get('/api/movie/trending');
      setTrendingMovie(response.data.content);
      console.log(`Trending Movie: ${response.data.content}`);
    };

    getTrendingMovie();
  }, []);

  return { trendingMovie };
};

export default useGetTrendingMovie;
