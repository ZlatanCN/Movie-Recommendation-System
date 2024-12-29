import express from 'express';
import {
  getMovieTrailers,
  getTrendingMovie,
  getMovieDetails,
  getSimilarMovies,
  getMoviesByCategory,
} from '../controllers/movieController.js';

const router = express.Router();

// @desc    Get trending movie
// @route   GET /api/movie/trending
router.get('/trending', getTrendingMovie);

// @desc    Get movie trailers
// @route   GET /api/movie/:movieId/trailers
router.get('/:movieId/trailers', getMovieTrailers);

// @desc    Get movie details
// @route   GET /api/movie/:movieId/details
router.get('/:movieId/details', getMovieDetails);

// @desc    Get similar movies
// @route   GET /api/movie/:movieId/similar
router.get('/:movieId/similar', getSimilarMovies);

// @desc    Get movies by category
// @route   GET /api/movie/:category
router.get('/:category', getMoviesByCategory);

export default router;