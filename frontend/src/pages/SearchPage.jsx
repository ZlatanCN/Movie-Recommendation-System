import { useState } from 'react';
import { motion } from 'framer-motion';
import NavBar from '../components/NavBar.jsx';
import { ConfigProvider, Input, message } from 'antd';
import { searchInputTheme } from '../theme/inputTheme.js';
import axios from 'axios';
import { LoadingOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { ORIGINAL_IMG_BASE_URL } from '../utils/constants.js';

const SearchPage = () => {
  const [searchType, setSearchType] = useState('movie');
  const [searchTitle, setSearchTitle] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();

  const handleSearch = async (e) => {
    e.preventDefault();
    setIsSearching(true);
    try {
      if (searchTitle.trim() === '') {
        const error = new Error('Search title cannot be empty!');
        error.status = 400;
        throw error;
      }
      const response = await axios.get(
        `/api/search/${searchType}/${searchTitle}`);
      setSearchResults(response.data.content);
    } catch (error) {
      let content;
      if (error.status === 400) {
        content = error.message || 'Error in searching';
      } else if (error.response.status === 404) {
        content = 'No results found';
      }
      messageApi.error({
        content: content,
        className: 'text-gray-300 font-bold font-mono text-[16px]',
      });
    } finally {
      setIsSearching(false);
    }
  };

  const handleTypeChange = (type) => {
    setSearchType(type);
    setSearchTitle('');
    setSearchResults([]);
  };

  return (
    <>
      {contextHolder}
      <div className={'bg-black min-h-screen text-white'}>
        <NavBar/>

        <div className={'container mx-auto px-4 py-8'}>
          {/* Search Type Buttons */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={'flex justify-center gap-3 mb-4'}
          >
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleTypeChange.bind(null, 'movie')}
              className={`py-2 px-4 rounded-lg ${searchType === 'movie'
                ? 'bg-red-600'
                : 'bg-gray-800'}`}
            >
              Movie
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleTypeChange.bind(null, 'person')}
              className={`py-2 px-4 rounded-lg ${searchType === 'person'
                ? 'bg-red-600'
                : 'bg-gray-800'}`}
            >
              Person
            </motion.button>
          </motion.section>

          {/* Search Form */}
          <motion.form
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            onSubmit={handleSearch}
            className={'flex gap-2 items-stretch mb-8 max-w-2xl mx-auto'}
          >
            <ConfigProvider theme={searchInputTheme}>
              <Input
                type={'text'}
                value={searchTitle}
                onChange={(e) => setSearchTitle(e.target.value)}
                placeholder={`Search ${searchType === 'movie'
                  ? 'Movies'
                  : 'People'}`}
                className={'w-full p-2 rounded-lg bg-gray-800 text-white border-gray-800 border-2'}
              />
            </ConfigProvider>
            <motion.button
              whileHover={{ scale: 1.02, backgroundColor: 'rgb(185 28 2)' }}
              whileTap={{ scale: 0.98 }}
              className={'bg-red-600 text-white p-2 rounded-lg'}
            >
              {isSearching ? <LoadingOutlined/> : 'Search'}
            </motion.button>
          </motion.form>

          {/* Search Results */}
          <section
            className={'grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4'}
          >
            {searchResults.map((result) => {
              if (!result?.poster_path && !result?.profile_path) {
                console.log('No image found for', result);
                return null;
              }
              console.log(result.profile_path);

              return (
                <div
                  key={result.id}
                  className={'p-4 rounded-lg'}
                >
                  {searchType === 'person' ? (
                    <Link
                      to={'/actor/' + result.name}
                      className={'flex flex-col items-center'}
                    >
                      <img
                        src={ORIGINAL_IMG_BASE_URL + result.profile_path}
                        alt={result.name}
                        className={'rounded-lg max-h-96 mx-auto'}
                      />
                      <h2 className={'mt-2 text-xl font-bold'}>
                        {result.name}
                      </h2>
                    </Link>
                  ) : (
                    <Link
                      to={'/watch/' + result.id}
                      className={'flex flex-col items-center'}
                    >
                      <img
                        src={ORIGINAL_IMG_BASE_URL + result.poster_path}
                        alt={result.title}
                        className={'w-full h-auto rounded-lg'}
                      />
                      <h2 className={'mt-2 font-semibold'}>
                        {result.title}
                      </h2>
                    </Link>
                  )}
                </div>
              )
            })}
          </section>
        </div>
      </div>
    </>
  );
};

export default SearchPage;
