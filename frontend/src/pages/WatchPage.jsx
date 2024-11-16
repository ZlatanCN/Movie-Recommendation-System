import { useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import axios from 'axios'
import NavBar from '../components/NavBar.jsx'
import {
  LeftOutlined,
  RightOutlined,
  StarOutlined,
} from '@ant-design/icons'
import ReactPlayer from 'react-player'
import { ORIGINAL_IMG_BASE_URL } from '../utils/constants.js'
import { motion } from 'framer-motion'
import useWatch from '../hooks/useWatch.jsx'
import formatDate from '../utils/formatDate.js'
import LoadingSpin from '../components/LoadingSpin.jsx'
import { Rate } from 'antd'
import RatingModal from '../components/RatingModal.jsx'
import SimilarMovies from '../components/SimilarMovies.jsx'

const WatchPage = () => {
  const { id } = useParams()
  const { trailers, content, isLoading } = useWatch(id)
  const [currentTrailerIndex, setCurrentTrailerIndex] = useState(0)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const showModal = () => {
    setIsModalOpen(true)
  }

  const rateMovie = async (id, rating) => {
    try {
      const response = await axios.post(`/api/rating/${id}`, { rating })
      if (!response.data.isSuccessful) {
        throw new Error('Failed to update rating!')
      }
    } catch (error) {
      console.error(error)
    }
  }

  const handleDirection = (direction) => {
    if (direction === 'left') {
      if (currentTrailerIndex > 0) {
        setCurrentTrailerIndex(currentTrailerIndex - 1)
      }
    } else if (direction === 'right') {
      if (currentTrailerIndex < trailers.length - 1) {
        setCurrentTrailerIndex(currentTrailerIndex + 1)
      }
    }
  }

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
    setCurrentTrailerIndex(0)
  }, [id])

  if (isLoading == null) {
    return (
      <LoadingSpin/>
    )
  }

  return (
    <>
      {content.id === undefined ? (
        <div className={'bg-black h-screen text-white'}>
          <div className={'max-w-6xl mx-auto'}>
            <NavBar/>
            <section className={'text-center mx-auto px-4 py-8 h-full mt-48'}>
              <h2 className={'text-2xl sm:text-5xl font-bold text-balance'}>
                Content not found 😢
              </h2>
            </section>
          </div>
        </div>
      ) : (
        <div className={'bg-black min-h-screen text-white'}>
          <div className={'mx-auto container px-4 py-8 h-full'}>
            {/* NavBar */}
            <NavBar/>

            {/* Movie Details */}
            <section
              className={'flex flex-col md:flex-row items-center justify-between gap-20 max-w-6xl mx-auto'}>
              <div className={'mb-4 md:mb-0'}>
                <h2 className={'text-5xl font-bold text-balance'}>
                  {content?.title || 'Movie Title'}
                </h2>
                <p className={'mt-2 text-lg'}>
                  {formatDate(content?.release_date)} | {' '}
                  {content?.adult ? (
                    <span className={'text-red-600'}>18+</span>
                  ) : (
                    <span className={'text-green-600'}>PG-13</span>
                  )}{' '}
                </p>
                <p className={'mt-4 text-lg'}>
                  {content?.overview}
                </p>
                <div className={'flex mt-8 text-xl items-center'}>
                  <span className={'font-semibold'}>
                    {parseFloat(content?.vote_average).toFixed(1)}
                  </span>
                  <Rate
                    disabled
                    allowHalf
                    defaultValue={content?.vote_average / 2}
                    className={'ml-3 flex justify-center items-center'}
                  />
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={showModal}
                    className={'bg-white text-black py-2 px-3 rounded ml-4 font-semibold text-lg flex items-center'}
                  >
                    <StarOutlined className={'pr-1.5'}/> Rate
                  </motion.button>
                  <RatingModal
                    content={content}
                    isModalOpen={isModalOpen}
                    setIsModalOpen={setIsModalOpen}
                    id={id}
                    updateRating={rateMovie}
                  />
                </div>
              </div>
              <img
                src={ORIGINAL_IMG_BASE_URL + content?.poster_path}
                alt={'poster'}
                className={'max-h-[600px] rounded-md'}
              />
            </section>

            {/* Trailer Video */}
            <section
              className={'relative aspect-video mb-8 p-2 sm:px-10 md:px-32 mt-16'}>
              {trailers.length > 0 ? (
                <>
                  <motion.div
                    key={currentTrailerIndex}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1, ease: 'easeInOut' }}
                  >
                    <ReactPlayer
                      controls={true}
                      width={'100%'}
                      height={'70vh'}
                      url={`https://www.youtube.com/watch?v=${trailers[currentTrailerIndex].key}`}
                      className={'mx-auto overflow-hidden rounded-lg'}
                    />
                  </motion.div>

                  {/* Trailers Buttons */}
                  {trailers.length > 0 && (
                    <>
                      <motion.button
                        className={`absolute left-0 top-1/2 transform -translate-y-1/2 bg-gray-500/70 text-white py-2 px-4 rounded-xl hover:bg-red-700 ${
                          currentTrailerIndex === 0
                            ? 'cursor-not-allowed opacity-50'
                            : 'opacity-70'}`}
                        whileTap={{ opacity: 0.5 }}
                        disabled={currentTrailerIndex === 0}
                        onClick={handleDirection.bind(this, 'left')}
                      >
                        <LeftOutlined/>
                      </motion.button>
                      <motion.button
                        className={`absolute right-0 top-1/2 transform -translate-y-1/2 bg-gray-500/70 text-white py-2 px-4 rounded-xl hover:bg-red-700 ${
                          currentTrailerIndex === trailers.length - 1
                            ? 'cursor-not-allowed opacity-50'
                            : 'opacity-70'}`}
                        whileTap={{ opacity: 0.5 }}
                        disabled={currentTrailerIndex === trailers.length - 1}
                        onClick={handleDirection.bind(this, 'right')}
                      >
                        <RightOutlined/>
                      </motion.button>
                    </>
                  )}
                </>
              ) : (
                <h2 className={'text-xl text-center mt-5'}>
                  No trailers available for{' '}
                  <span className={'font-bold text-red-600'}>
                    {content?.title || 'this movie'}
                  </span>
                </h2>
              )}
            </section>

            {/* Similar Movies */}
            <SimilarMovies movieId={id}/>
          </div>
        </div>
      )}
    </>
  )
}

export default WatchPage