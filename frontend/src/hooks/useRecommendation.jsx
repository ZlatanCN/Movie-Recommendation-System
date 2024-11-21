import { useEffect, useState } from 'react';
import ColorThief from 'colorthief';
import axios from 'axios';

const useRecommendation = () => {
  const [recommendedMovies, setRecommendedMovies] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [dominantColor, setDominantColor] = useState(null);
  const startIndex = currentPage - 1;
  const currentMovie = recommendedMovies.slice(startIndex, startIndex + 1);

  const handleDirection = (direction) => {
    if (direction === 'prev') {
      setCurrentPage((prev) => Math.max(prev - 1, 1));
    } else if (direction === 'next') {
      setCurrentPage((prev) => Math.min(prev + 1, recommendedMovies.length));
    }
  }

  const fetchRecommendations = async () => {
    try {
      const response = await axios.get(
        `/api/recommendation/collaborative`);
      if (response.data.isSuccessful) {
        setRecommendedMovies(response.data.content);
        console.log('Recommendations fetched successfully');
        console.log(response.data.content);
      } else {
        console.log('Error fetching recommendations');
        setRecommendedMovies([]);
      }
    } catch {
      console.log('Error fetching recommendations');
      setRecommendedMovies([]);
    }

    // setRecommendedMovies([
    //   {
    //     id: 238,
    //     title: 'The Godfather',
    //     overview: 'Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone barely survives an attempt on his life, his youngest son, Michael steps in to take care of the would-be killers, launching, a campaign of bloody revenge.',
    //     vote_average: 8.7,
    //     poster_path: '/3bhkrj58Vtu7enYsRolD1fZdja1.jpg',
    //     backdrop_path: '/ejdD20cdHNFAYAN2DlqPToXKyzx.jpg',
    //     recommendation: 0.9,
    //   }, {
    //     id: 424,
    //     title: 'Schindler\'s List',
    //     overview: 'The true story of how businessman Oskar Schindler saved over a thousand Jewish lives from',
    //     vote_average: 8.6,
    //     poster_path: '/c8Ass7acuOe4za6DhSattE359gr.jpg',
    //     backdrop_path: "/zb6fM1CX41D9rF9hdgclu0peUmy.jpg",
    //     recommendation: 0.8,
    //   }, {
    //     id: 680,
    //     title: 'Pulp Fiction',
    //     overview: 'A burger-loving hit, He and his partner Jules Winnfield have a penchant for philosophical discussions. But is he ready to call it quits?',
    //     vote_average: 8.5,
    //     poster_path: "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
    //     backdrop_path: "/suaEOtk1N1sgg2MTM7oZd2cfVp3.jpg",
    //     recommendation: 0.7,
    //   }, {
    //     id: 129,
    //     title: 'Spirited Away',
    //     overview: 'Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone barely survives an attempt on his life, his youngest son, Michael steps in to take care of the would-be killers, launching, a campaign of bloody revenge.',
    //     vote_average: 8.5,
    //     poster_path: '/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg',
    //     backdrop_path: '/mSDsSDwaP3E7dEfUPWy4J0djt4O.jpg',
    //     recommendation: 0.6,
    //   }]);
  };

  useEffect(() => {
    fetchRecommendations();
  }, []);

  useEffect(() => {
    if (currentMovie.length > 0) {
      const img = new Image();
      img.crossOrigin = 'Anonymous';
      img.src = `https://image.tmdb.org/t/p/w200${currentMovie[0].poster_path}`;
      img.onload = () => {
        const colorThief = new ColorThief();
        const dominant = colorThief.getColor(img);
        setDominantColor(`rgb(${dominant.join(',')})`);
      };
    }
  }, [currentMovie]);

  return {
    recommendedMovies,
    currentMovie,
    currentPage,
    dominantColor,
    handleDirection,
  };
}

export default useRecommendation;