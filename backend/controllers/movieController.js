import chalk from 'chalk';
import fetchFromTMDB from '../config/tmdb.js';

const getTrendingMovie = async (req, res) => {
  try {
    const data = await fetchFromTMDB('https://api.themoviedb.org/3/trending/movie/week');
    const randomMovie = data.results[Math.floor(
      Math.random() * data.results?.length)];

    res.status(200).json({
      isSuccessful: true,
      content: randomMovie,
    });
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in getTrendingMovie - movieController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      error: 'Internal Server Error',
    });
  }
};

const getMovieTrailers = async (req, res) => {
  const { movieId } = req.params;

  try {
    const data = await fetchFromTMDB(
      `https://api.themoviedb.org/3/movie/${movieId}/videos`);
    res.status(200).json({
      isSuccessful: true,
      trailers: data.results,
    });
  } catch (error) {
    if (error.message.includes('404')) {
      return res.status(404).send(null);
    } else {
      console.log(chalk.red.bold(
        `Error in getMovieTrailers - movieController: ${error.message}`));
      res.status(500).json({
        isSuccessful: false,
        message: 'Internal Server Error',
      });
    }
  }
};

const getMovieDetails = async (req, res) => {
  const { movieId } = req.params;

  try {
    const data = await fetchFromTMDB(
      `https://api.themoviedb.org/3/movie/${movieId}`);
    res.status(200).json({
      isSuccessful: true,
      content: data,
    });
  } catch (error) {
    if (error.message.includes('404')) {
      return res.status(404).send(null);
    } else {
      console.log(chalk.red.bold(
        `Error in getMovieTrailers - movieController: ${error.message}`));
      res.status(500).json({
        isSuccessful: false,
        message: 'Internal Server Error',
      });
    }
  }
};

const getSimilarMovies = async (req, res) => {
  const { movieId } = req.params;

  try {
    const data = await fetchFromTMDB(
      `https://api.themoviedb.org/3/movie/${movieId}/similar`);
    res.status(200).json({
      isSuccessful: true,
      similar: data.results,
    });
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in getSimilarMovies - movieController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal Server Error',
    });
  }
};

const getMoviesByCategory = async (req, res) => {
  const { category } = req.params;

  try {
    const data = await fetchFromTMDB(
      `https://api.themoviedb.org/3/movie/${category}`);
    res.status(200).json({
      isSuccessful: true,
      content: data.results,
    });
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in getMoviesByCategory - movieController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal Server Error',
    });
  }
};

export {
  getTrendingMovie,
  getMovieTrailers,
  getMovieDetails,
  getSimilarMovies,
  getMoviesByCategory,
};