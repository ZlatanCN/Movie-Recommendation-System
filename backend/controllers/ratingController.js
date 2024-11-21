import chalk from 'chalk';
import User from '../models/userModel.js';
import fetchFromTMDB from '../config/tmdb.js';
import { getMySQLPool } from '../config/db.js';

const rateMovie = async (req, res) => {
  const { movieId } = req.params;
  const { rating } = req.body;
  const userIntId = req.user.intId;

  try {
    const data = await fetchFromTMDB(
      `https://api.themoviedb.org/3/movie/${movieId}`);
    const userId = req.userId;
    const isAlreadyRated = req.user.ratedMovies.some(
      (movie) => (movie.id === movieId),
    );

    if (!isAlreadyRated) {
      await User.findByIdAndUpdate(userId, {
        $push: {
          ratedMovies: {
            id: movieId,
            title: data.title,
            image: data.poster_path,
            rating: rating,
            createdAt: new Date(),
            updatedAt: new Date(),
          },
        },
      });
    } else {
      await User.updateOne(
        { _id: userId, 'ratedMovies.id': movieId },
        {
          $set: {
            'ratedMovies.$.rating': rating,
            'ratedMovies.$.updatedAt': new Date(),
          },
        },
      );
    }

    // Connect to MySQL database
    const pool = getMySQLPool();

    // Check if the rowId exists, if true, update the rating, else insert a new row
    const checkRatingQuery = `SELECT *
                              FROM ratings
                              WHERE userId = ?
                                AND tmdbId = ?`;
    const [checkRating] = await pool.query(checkRatingQuery,
      [userIntId, movieId]);
    if (checkRating.length > 0) {
      const updateRatingQuery = `UPDATE ratings
                                 SET rating = ?
                                 WHERE rowId = ?`;
      await pool.query(updateRatingQuery, [rating, checkRating[0].rowId]);
    } else {
      const insertRatingQuery = `INSERT INTO ratings (userId, tmdbId, rating)
                                 VALUES (?, ?, ?)`;
      await pool.query(insertRatingQuery, [userIntId, movieId, rating]);
    }

    // SELECT the last 2 rows to check if the data is saved
    const [rows] = await pool.query(
      'SELECT * FROM ratings ORDER BY rowId DESC LIMIT 2');
    console.log(rows);

    res.status(200).json({
      isSuccessful: true,
      message: 'Movie rated successfully',
    });
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in rateMovie - ratingController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal Server Error',
    });
  }
};

const getRatedMovies = async (req, res) => {
  try {
    const userId = req.userId;
    const user = req.user;
    const ratedMovies = user.ratedMovies;
    res.status(200).json({
      isSuccessful: true,
      content: ratedMovies,
    });
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in getRatedMovies - ratingController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal Server Error',
    });
  }
};

const deleteRating = async (req, res) => {
  const userId = req.userId;
  let { movieId } = req.params;
  const userIntId = req.user.intId;

  try {
    await User.findByIdAndUpdate(userId, {
      $pull: {
        ratedMovies: { id: movieId },
      },
    });

    // Connect to MySQL database
    const pool = getMySQLPool();

    // Delete the rating from the MySQL database
    const deleteRatingQuery = `DELETE
                               FROM ratings
                               WHERE userId = ?
                                 AND tmdbId = ?`;
    await pool.query(deleteRatingQuery, [userIntId, movieId]);

    // SELECT the last 2 rows to check if the data is saved
    const [rows] = await pool.query(
      'SELECT * FROM ratings ORDER BY rowId DESC LIMIT 2');
    console.log(rows);

    res.status(200).json({
      isSuccessful: true,
      message: 'Rating deleted successfully',
    });
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in deleteRating - ratingController: ${error.message}`));
    res.status(500).json({
      isSuccessful: false,
      message: 'Internal Server Error',
    });
  }
};

export { rateMovie, getRatedMovies, deleteRating };