import { Link } from 'react-router-dom'
import { SMALL_IMG_BASE_URL } from '../utils/constants.js'
import { motion } from 'framer-motion'
import { LeftOutlined, LoadingOutlined, RightOutlined } from '@ant-design/icons'
import useSimilarMovies from '../hooks/useSimilarMovies.jsx'
import PropTypes from 'prop-types'
import { Skeleton } from 'antd'
import LoadingSpin from './LoadingSpin.jsx'

const SimilarMovies = (props) => {
  const {
    similarMovies,
    sliderRef,
    isLoading,
  } = useSimilarMovies(props.movieId)

  const scrollX = (direction) => {
    if (sliderRef.current) {
      const distance = -sliderRef.current.offsetWidth
      if (direction === 'left') {
        sliderRef.current.scrollBy({ left: distance, behavior: 'smooth' })
      } else if (direction === 'right') {
        sliderRef.current.scrollBy({ left: -distance, behavior: 'smooth' })
      }
    }
  }

  return (
    <section className={'mt-12 max-w-5xl mx-auto relative'}>
      <h3 className={'text-3xl font-bold mb-4'}>
        Similar Movies
      </h3>
      <div
        className={'flex overflow-x-scroll scrollbar-hide gap-4 pb-4 group'}
        ref={sliderRef}
      >
        {isLoading ? (
          <LoadingOutlined
            className={'text-4xl text-red-600 mx-auto mt-24 mb-24'}
          />
        ) : (
          <>
            {similarMovies.map((content) => {
              if (!content?.poster_path) return null
              return (
                <Link
                  key={content.id}
                  to={`/watch/${content.id}`}
                  className={'w-52 flex-none'}
                >
                  <img
                    src={SMALL_IMG_BASE_URL + content?.poster_path}
                    alt={content.title}
                    className={'w-full h-auto rounded-md'}
                  />
                  <h4 className={'mt-2 text-lg font-semibold'}>
                    {content.title}
                  </h4>
                </Link>
              )
            })}

            <motion.button
              onClick={scrollX.bind(this, 'left')}
              whileTap={{ opacity: 0.5 }}
              className={'absolute top-[45%] -translate-y-1/2 -left-16 flex items-center justify-center size-12 rounded-full bg-gray-800 bg-opacity-50 hover:bg-opacity-60 text-white z-10 hover:bg-red-600'}>
              <LeftOutlined/>
            </motion.button>
            <motion.button
              onClick={scrollX.bind(this, 'right')}
              whileTap={{ opacity: 0.5 }}
              className={'absolute top-[45%] -translate-y-1/2 -right-16 flex items-center justify-center size-12 rounded-full bg-gray-800 bg-opacity-50 hover:bg-opacity-60 text-white z-10 hover:bg-red-600'}>
              <RightOutlined/>
            </motion.button>
          </>
        )}
      </div>
    </section>

  )
}

SimilarMovies.propTypes = {
  movieId: PropTypes.string.isRequired,
}

export default SimilarMovies