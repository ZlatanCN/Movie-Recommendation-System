import express from 'express';
import {
  deleteRating,
  getRatedMovies,
  rateMovie,
} from '../controllers/ratingController.js';

const router = express.Router();

// @desc    Rate a movie
// @route   POST /api/rating/:movieId
router.post('/:movieId', rateMovie);

// @desc    Get all rated movies
// @route   GET /api/rating
router.get('/', getRatedMovies);

// @desc    Delete a rating for a movie
// @route   DELETE /api/rating/:movieId
router.delete('/:movieId', deleteRating);

export default router;