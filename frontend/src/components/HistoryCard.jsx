import PropTypes from 'prop-types'
import { SMALL_IMG_BASE_URL } from '../utils/constants.js'
import formatDate from '../utils/formatDate.js'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { DeleteOutlined } from '@ant-design/icons'

const HistoryCard = (props) => {
  return (
    <div
      key={props.item.id}
      className={'bg-gray-900 p-4 rounded-lg flex items-start'}
    >
      <Link to={`/watch/${props.item.id}`}>
        <img
          src={SMALL_IMG_BASE_URL + props.item.image}
          alt={props.item.title || props.item.name || 'Image'}
          className={'size-16 rounded-lg object-cover mr-4'}
        />
      </Link>
      <div className={'flex flex-col'}>
        <span className={'text-white text-lg'}>
          {props.item.title}
        </span>
        <span className={'text-gray-400 text-sm'}>
          {formatDate(props.item.createdAt)}
        </span>
      </div>
      <span className={`py-1 px-3 min-w-20 text-center rounded-full text-sm ml-auto font-semibold 
        ${props.item.searchType === 'movie'
        ? 'bg-red-600'
        : 'bg-green-600'}`}
      >
        {props.item.searchType.toUpperCase()}
      </span>
      <motion.button
        whileHover={{ scale: 1.1, color: 'rgb(220 38 38 / 1)' }}
        whileTap={{ scale: 0.9 }}
        onClick={props.handleDelete.bind(this, props.item)}
        className={'size-6 ml-4 cursor-pointer rounded-full flex items-center justify-center'}
      >
        <DeleteOutlined/>
      </motion.button>
    </div>
  )
}

HistoryCard.propTypes = {
  item: PropTypes.object.isRequired,
  handleDelete: PropTypes.func.isRequired,
}

export default HistoryCard