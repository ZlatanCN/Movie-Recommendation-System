import chalk from 'chalk';
import fetchFromTMDB from '../config/tmdb.js';
import User from '../models/userModel.js';

const searchPerson = async (req, res) => {
  const { query } = req.params;

  try {
    const response = await fetchFromTMDB(
      `https://api.themoviedb.org/3/search/person?query=${query}`);

    if (response.results.length === 0) {
      return res.status(404).send(null);
    } else {
      const userId = req.userId;
      await User.findByIdAndUpdate(userId, {
        $push: {
          searchHistory: {
            id: response.results[0].id,
            image: response.results[0].profile_path,
            title: response.results[0].name,
            searchType: 'person',
            createdAt: new Date(),
          },
        },
      });

      res.status(200).json({
        isSuccessful: true,
        content: response.results,
      });
    }
  } catch (error) {
    console.log(
      chalk.red.bold(
        `Error in searchPerson - searchController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal server error',
    });
  }
};

const searchMovie = async (req, res) => {
  const { query } = req.params;

  try {
    const response = await fetchFromTMDB(
      `https://api.themoviedb.org/3/search/movie?query=${query}`);

    if (response.results.length === 0) {
      return res.status(404).send(null);
    } else {
      const userId = req.userId;
      await User.findByIdAndUpdate(userId, {
        $push: {
          searchHistory: {
            id: response.results[0].id,
            image: response.results[0].poster_path,
            title: response.results[0].title,
            searchType: 'movie',
            createdAt: new Date(),
          },
        },
      });

      res.status(200).json({
        isSuccessful: true,
        content: response.results,
      });
    }
  } catch (error) {
    console.log(
      chalk.red.bold(
        `Error in searchMovie - searchController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal server error',
    });
  }
};

const getSearchHistory = async (req, res) => {
  try {
    const user = req.user;

    res.status(200).json({
      isSuccessful: true,
      content: user.searchHistory,
    });
  } catch (error) {
    console.log(
      chalk.red.bold(
        `Error in getSearchHistory - searchController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal server error',
    });
  }
};

const deleteOneSearchHistory = async (req, res) => {
  const userId = req.userId;
  let { id } = req.params;
  id = parseInt(id);

  try {
    await User.findByIdAndUpdate(userId, {
      $pull: {
        searchHistory: { id: id },
      },
    });

    res.status(200).json({
      isSuccessful: true,
      message: 'Search history deleted successfully',
    });
  } catch (error) {
    console.log(
      chalk.red.bold(
        `Error in deleteOneSearchHistory - searchController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal server error',
    });
  }
};

export { searchPerson, searchMovie, getSearchHistory, deleteOneSearchHistory };