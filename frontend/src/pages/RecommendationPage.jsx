import NavBar from '../components/NavBar.jsx'
import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { Rate } from 'antd'
import ColorThief from 'colorthief'
import { LeftOutlined, RightOutlined } from '@ant-design/icons'
import { SMALL_IMG_BASE_URL } from '../utils/constants.js'
import { Link } from 'react-router-dom'
import axios from 'axios'

const RecommendationPage = () => {
  const [recommendedMovies, setRecommendedMovies] = useState([])
  const [currentPage, setCurrentPage] = useState(1)
  const [dominantColor, setDominantColor] = useState(null)
  const startIndex = currentPage - 1
  const currentMovie = recommendedMovies.slice(startIndex, startIndex + 1)

  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        const response = await axios.get(
          `/api/recommendation/collaborative/${1}`)
        if (response.data.isSuccessful) {
          setRecommendedMovies(response.data.content)
        } else {
          console.log('Error fetching recommendations')
          setRecommendedMovies([])
        }
      } catch {
        console.log('Error fetching recommendations')
        setRecommendedMovies([])
      }
    }
    fetchRecommendations()
  }, [])

  useEffect(() => {
    if (currentMovie.length > 0) {
      const img = new Image()
      img.crossOrigin = 'Anonymous'
      img.src = `https://image.tmdb.org/t/p/w200${currentMovie[0].poster_path}`
      img.onload = () => {
        const colorThief = new ColorThief()
        const dominant = colorThief.getColor(img)
        setDominantColor(`rgb(${dominant.join(',')})`)
      }
    }
  }, [currentMovie])

  return (
    <div className={'bg-black text-white min-h-screen relative'}>
      <NavBar/>

      <section className={'max-w-6xl mx-auto px-4 py-8'}>
        <h1 className={'text-3xl font-bold mb-8'}>
          Your Recommendations
        </h1>

        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className={'grid grid-cols-1 min-h-[70vh]'}
        >
          {currentMovie.map((movie) => (
            <motion.div
              key={movie.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className={'flex justify-center items-center rounded-2xl'}
              style={{
                background: dominantColor,
              }}
            >
              {/* Movie Poster */}
              <img
                src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`}
                alt={movie.title}
                className={'w-[30rem] h-[45rem] rounded-l-2xl self-start'}
              />
              <section
                className={'self-start flex flex-col justify-between min-h-[45rem] w-full relative'}>
                <img
                  src={SMALL_IMG_BASE_URL + movie.backdrop_path}
                  alt={movie.title}
                  className={'absolute w-full h-full object-cover opacity-30 rounded-r-2xl -z-10 mr-3 '}
                />
                <h2
                  className={'text-5xl font-semibold rounded-lg px-6 py-6'}
                >
                  {movie.title}
                </h2>
                <article
                  className={'text-xl text-gray-200 rounded-lg px-6 py-6'}
                >
                  {movie.overview}
                </article>

                <section className={'flex justify-between'}>
                  {/* Movie Rating */}
                  <section
                    className={'flex flex-col gap-2 rounded-lg px-6 py-6 w-fit relative'}
                  >
                    <h3
                      className={'text-3xl font-semibold mx-auto'}
                      // style={{ textShadow: '2px 2px 4px rgba(0, 0, 0, 0.38)' }}
                    >
                      {(movie.vote_average).toFixed(1)} / 10
                    </h3>
                    <Rate
                      allowHalf
                      style={{
                        fontSize: 30,
                      }}
                      value={movie.vote_average / 2}
                      disabled
                    />
                  </section>

                  {/*Buttons*/}
                  <section className={'flex justify-center gap-6 mr-6'}>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={'bg-gray-200 text-black font-semibold rounded-lg h-fit px-3 py-2 self-end mb-6'}
                    >
                      <Link to={`/watch/${movie.id}`}>
                        Watch
                      </Link>
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={'bg-red-700 text-white font-semibold rounded-lg h-fit px-3 py-2 self-end mb-6'}
                    >
                      Rate
                    </motion.button>
                  </section>
                </section>

                {/* Pagination */}
                <section className={'flex justify-between mb-1'}>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    disabled={currentPage === 1}
                    onClick={() => setCurrentPage(
                      (prev) => Math.max(prev - 1, 1))}
                    className={`text-white px-4 py-2 rounded-lg font-semibold ${currentPage ===
                    1 && 'cursor-not-allowed opacity-50'}`}
                  >
                    <LeftOutlined/>
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    disabled={currentPage === recommendedMovies.length}
                    onClick={() => setCurrentPage(
                      (prev) => Math.min(prev + 1, recommendedMovies.length))}
                    className={`text-white px-4 py-2 rounded-lg font-semibold ${currentPage ===
                    recommendedMovies.length &&
                    'cursor-not-allowed opacity-50'}`}
                  >
                    <RightOutlined/>
                  </motion.button>
                </section>
              </section>
            </motion.div>
          ))}
        </motion.div>
      </section>
    </div>
  )
}

export default RecommendationPage