import chalk from 'chalk';
import User from '../models/userModel.js';
import fetchFromTMDB from '../config/tmdb.js';

const rateMovie = async (req, res) => {
  const { movieId } = req.params;
  const { rating } = req.body;

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

  try {
    await User.findByIdAndUpdate(userId, {
      $pull: {
        ratedMovies: { id: movieId },
      },
    });

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