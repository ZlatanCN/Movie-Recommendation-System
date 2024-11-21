import PropTypes from 'prop-types';
import { ORIGINAL_IMG_BASE_URL } from '../utils/constants.js';
import { ConfigProvider, Popconfirm, Rate } from 'antd';
import { motion } from 'framer-motion';
import { logoutPopConfirmTheme } from '../theme/popConfirmTheme.js';
import { QuestionCircleOutlined } from '@ant-design/icons';
import RatingModal from './RatingModal.jsx';
import { useState } from 'react';
import { Link } from 'react-router-dom';

const MovieRating = (props) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div
      key={props.movie.id}
      className={'bg-gray-900 p-4 rounded-lg flex items-start max-h-[375px] min-h-[375px] flex-col overflow-hidden'}
    >
      {/*Movie Image and Title*/}
      <section className={'flex items-end ml-8 w-full justify-between'}>
        <Link to={`/watch/${props.movie.id}`}>
          <img
            src={ORIGINAL_IMG_BASE_URL + props.movie.image}
            alt={props.movie.title || props.movie.name || 'Image'}
            className={'rounded-lg object-cover mr-4 max-w-[180px]'}
          />
        </Link>

        {/*Movie Rating*/}
        <div className={'flex flex-col justify-center items-center mr-16'}>
          <p className={'text-white text-4xl font-semibold'}>
            {props.movie.rating * 2} / 10
          </p>
          <Rate
            allowHalf
            style={{
              fontSize: 36,
            }}
            value={props.movie.rating}
            disabled
            className={'flex justify-center items-center mt-8 bg-gray-700/20 rounded-lg py-4 px-2'}
          />

          {/*Edit and Delete Buttons*/}
          <div className={'mt-10 flex self-end gap-4'}>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={'bg-red-600 rounded-lg px-4 py-2 font-semibold self-end'}
              onClick={() => setIsModalOpen(true)}
            >
              Edit
            </motion.button>
            <RatingModal
              content={props.movie}
              id={props.movie.id}
              isModalOpen={isModalOpen}
              setIsModalOpen={setIsModalOpen}
              updateRating={props.updateRating}
            />
            <ConfigProvider theme={logoutPopConfirmTheme}>
              <Popconfirm
                title={'Are you sure you want to delete this rating?'}
                onConfirm={props.handleDelete.bind(this, props.movie)}
                okText={<span className={'font-semibold'}>Yes</span>}
                cancelText={<span className={'font-semibold'}>No</span>}
                color={'rgb(31 41 55)'}
                icon={
                  <QuestionCircleOutlined
                    style={{
                      color: 'rgb(220 38 38 / 1)',
                    }}
                  />
                }
              >
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={'bg-red-600 rounded-lg px-4 py-2 font-semibold self-end'}
                >
                  Delete
                </motion.button>
              </Popconfirm>
            </ConfigProvider>
          </div>
        </div>
      </section>

      {/*Movie Title*/}
      <p className={'ml-8 font-semibold mt-6 max-w-[180px]'}>
        {props.movie.title || props.movie.name}
      </p>
    </div>
  );
};

MovieRating.propTypes = {
  movie: PropTypes.object.isRequired,
  handleDelete: PropTypes.func.isRequired,
  updateRating: PropTypes.func.isRequired,
};

export default MovieRating;
