import express from 'express';
import {
  searchPerson,
  searchMovie,
  getSearchHistory,
  deleteOneSearchHistory,
} from '../controllers/searchController.js';

const router = express.Router();

// @desc    Search for a person
// @route   GET /api/search/person/:query
router.get('/person/:query', searchPerson);

// @desc    Search for a movie
// @route   GET /api/search/movie/:query
router.get('/movie/:query', searchMovie);

// @desc    Get search history
// @route   GET /api/search/history
router.get('/history', getSearchHistory);

// @desc    Delete one search history
// @route   DELETE /api/search/history/:id
router.delete('/history/:id', deleteOneSearchHistory);

export default router;