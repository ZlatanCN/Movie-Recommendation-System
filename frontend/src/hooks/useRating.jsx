import { useEffect, useState } from 'react';
import axios from 'axios';

const useRating = () => {
  const [ratedmovies, setRatedmovies] = useState([]);

  const fetchRatedMovies = async () => {
    try {
      const response = await axios.get('/api/rating');
      setRatedmovies(response.data.content);
    } catch (error) {
      console.error(error);
    }
  }

  useEffect(() => {
    fetchRatedMovies()
  }, []);

  return { ratedmovies, setRatedmovies };
}

export default useRating;