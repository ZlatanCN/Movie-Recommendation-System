import NavBar from '../components/NavBar.jsx';
import { motion } from 'framer-motion';
import useHistory from '../hooks/useHistory.jsx';
import LoadingSpin from '../components/LoadingSpin.jsx';
import axios from 'axios';
import HistoryCard from '../components/HistoryCard.jsx'

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
              <HistoryCard handleDelete={handleDelete} item={item}/>
            ))}
          </motion.div>
        )}
      </section>
    </div>
  );
};

export default HistoryPage;
