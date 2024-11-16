import { useEffect, useState } from 'react';
import axios from 'axios';

const useRating = () => {
  const [ratedmovies, setRatedmovies] = useState([]);

  const fetchRatedMovies = async () => {
    try {
      const response = await axios.get('/api/rating');
      const sortedMovies = response.data.content.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
      setRatedmovies(sortedMovies);
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