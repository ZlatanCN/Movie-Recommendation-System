import { useEffect, useState } from 'react';
import axios from 'axios';

const useWatch = (id) => {
  const [trailers, setTrailers] = useState([]);
  const [content, setContent] = useState({});
  const [isLoading, setIsLoading] = useState();

  const getMovieTrailers = async () => {
    try {
      const response = await axios.get(`/api/movie/${id}/trailers`);
      setTrailers(response.data.trailers);
    } catch (error) {
      if (error.message.includes('404')) {
        setTrailers([]);
      }
      console.log(`Error in getMovieTrailers - WatchPage: ${error.message}`);
    }
  };

  const getMovieDetails = async () => {
    try {
      const response = await axios.get(`/api/movie/${id}/details`);
      setContent(response.data.content);
    } catch (error) {
      if (error.message.includes('404')) {
        setContent({});
      }
      console.log(`Error in getMovieDetails - WatchPage: ${error.message}`);
    }
  }

  useEffect(() => {
    const fetchTrailersAndSimilar = async () => {
      await Promise.all([getMovieTrailers(), getMovieDetails()]);
      setIsLoading(false);
    };

    fetchTrailersAndSimilar();
  }, [id]);

  return { trailers, content, isLoading };
}

export default useWatch;