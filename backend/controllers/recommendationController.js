import { PythonShell } from 'python-shell'
import chalk from 'chalk'
import fetchFromTMDB from '../config/tmdb.js'
import axios from 'axios'

const recommendContentBased = async (req, res) => {
  const { id } = req.params

  try {
    const response = await axios.get(
      `http://localhost:6000/api/recommendation/${id}`)

    if (response.data.isSuccessful) {
      console.log(response.data.content)
      const movieEntries = Object.entries(response.data.content)
      const structuredData = await Promise.all(
        movieEntries.map(
          async ([id, similarity]) => {
            const movieDetails = await fetchFromTMDB(
              `https://api.themoviedb.org/3/movie/${id}`)

            return {
              id: movieDetails.id,
              title: movieDetails.title || movieDetails.original_title ||
                movieDetails.name,
              poster_path: movieDetails.poster_path,
              similarity: similarity,
            }
          },
        ),
      )

      res.status(200).json({
        isSuccessful: true,
        content: structuredData,
      })
    } else {
      const data = await fetchFromTMDB(
        `https://api.themoviedb.org/3/movie/${id}/similar`)

      res.status(200).json({
        isSuccessful: true,
        content: data.results,
      })
    }
  } catch (error) {
    console.log(chalk.red.bold(
      `Error in recommendContentBased - recommendationController: ${error.message}`))
    res.status(500).json({
      isSuccessful: false,
      message: error.message || 'Internal server error',
    })
  }
}

export { recommendContentBased }
