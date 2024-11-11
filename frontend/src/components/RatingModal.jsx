import PropTypes from 'prop-types';
import { rateModalTheme } from '../theme/modalTheme.js';
import { ConfigProvider, Modal, Rate } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';
import axios from 'axios';
import { useState } from 'react';

const RatingModal = (props) => {
  const [rate, setRate] = useState(0);
  const [isRating, setIsRating] = useState(false);

  const handleOk = async () => {
    try {
      setIsRating(true);
      await props.updateRating(props.id, rate);
    } catch (error) {
      console.error('Error in rating movie:', error.message);
    } finally {
      props.setIsModalOpen(false);
      setIsRating(false);
      setRate(0);
    }
  };

  const handleCancel = () => {
    props.setIsModalOpen(false);
    setRate(0);
  };

  return (
    <ConfigProvider theme={rateModalTheme}>
      <Modal
        title={
          <div>
            <span className={'text-red-500'}>Rate </span>
            {props.content?.title}
          </div>
        }
        open={props.isModalOpen}
        onOk={handleOk}
        okText={isRating ? (
          <LoadingOutlined className={'size-3'}/>
        ) : (
          <span className={'font-semibold'}>OK</span>
        )}
        cancelText={<span className={'font-semibold'}>Cancel</span>}
        onCancel={handleCancel}
      >
        <Rate
          allowHalf
          style={{
            fontSize: 36,
          }}
          value={rate}
          onChange={(value) => setRate(value)}
          className={'flex justify-center items-center mt-8 bg-gray-700/20 rounded-lg py-4 mb-6'}
        />
      </Modal>
    </ConfigProvider>
  );
};

RatingModal.propTypes = {
  content: PropTypes.object.isRequired,
  isModalOpen: PropTypes.bool.isRequired,
  setIsModalOpen: PropTypes.func.isRequired,
  id: PropTypes.string.isRequired,
  updateRating: PropTypes.func.isRequired,
};

export default RatingModal;
