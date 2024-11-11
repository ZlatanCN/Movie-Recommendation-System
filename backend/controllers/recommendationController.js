import { PythonShell } from 'python-shell'
import chalk from 'chalk'
import fetchFromTMDB from '../config/tmdb.js'

const recommendContentBased = async (req, res) => {
  const { id } = req.params

  let options = {
    mode: 'text',
    pythonOptions: ['-u'],
    args: [id],
  }

  try {
    const message = await PythonShell.run(
      'algorithm/contentbased_recommendation.py', options)
    if (message[0].startsWith('Error:')) {
      throw new Error(message)
    } else {
      const recommendations = JSON.parse(message[0])
      const formattedMessage = await Promise.all(
        Object.entries(recommendations).map(async ([id, score]) => {
          const data = await fetchFromTMDB(`https://api.themoviedb.org/3/movie/${id}`)
          return {
            id: parseInt(id),
            score: score,
            details: data,
          }
        }),
      )
      res.status(200).json({
        isSuccessful: true,
        content: formattedMessage,
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
