import NavBar from '../components/NavBar.jsx';
import { SMALL_IMG_BASE_URL } from '../utils/constants.js';
import formatDate from '../utils/formatDate.js';
import { motion } from 'framer-motion';
import useHistory from '../hooks/useHistory.jsx';
import LoadingSpin from '../components/LoadingSpin.jsx';
import { DeleteOutlined } from '@ant-design/icons';
import axios from 'axios';

const HistoryPage = () => {
  const { searchHistory, isLoading, setSearchHistory } = useHistory();

  const handleDelete = async (item) => {
    try {
      await axios.delete(`/api/search/history/${item.id}`);
      setSearchHistory(searchHistory.filter((history) => (history.id !== item.id)));
    } catch (error) {
      console.error(error);
    }
  }

  if (isLoading) {
    return <LoadingSpin/>;
  }

  return (
    <div className={'bg-black min-h-screen text-white'}>
      <NavBar/>
      <section className={'max-w-6xl mx-auto px-4 py-8'}>
        <h1 className={'text-3xl font-bold mb-8'}>
          Search History
        </h1>
        {searchHistory.length === 0 ? (
          <div className={'flex justify-center items-center h-96'}>
            <p className={'text-xl'}>
              {/*No search history found*/}
            </p>
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={'grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4'}
          >
            {searchHistory.map((item) => (
              <div
                key={item.id}
                className={'bg-gray-900 p-4 rounded-lg flex items-start'}
              >
                <img
                  src={SMALL_IMG_BASE_URL + item.image}
                  alt={item.title || item.name || 'Image'}
                  className={'size-16 rounded-lg object-cover mr-4'}
                />
                <div className={'flex flex-col'}>
                  <span className={'text-white text-lg'}>
                    {item.title}
                  </span>
                  <span className={'text-gray-400 text-sm'}>
                    {formatDate(item.createdAt)}
                  </span>
                </div>
                <span className={`py-1 px-3 min-w-20 text-center rounded-full text-sm ml-auto font-semibold 
                ${item.searchType === 'movie'
                  ? 'bg-red-600'
                  : 'bg-green-600'}`}
                >
                  {item.searchType.toUpperCase()}
                </span>
                <motion.button
                  whileHover={{ scale: 1.1, color: 'rgb(220 38 38 / 1)' }}
                  whileTap={{ scale: 0.9 }}
                  onClick={handleDelete.bind(this, item)}
                  className={'size-6 ml-4 cursor-pointer rounded-full flex items-center justify-center'}
                >
                  <DeleteOutlined />
                </motion.button>
              </div>
            ))}
          </motion.div>
        )}
      </section>
    </div>
  );
};

export default HistoryPage;
