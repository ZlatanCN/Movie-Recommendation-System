import { useEffect, useState } from 'react';
import axios from 'axios';

const useHistory = () => {
  const [searchHistory, setSearchHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const getSearchHistory = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get('/api/search/history');
      setSearchHistory(response.data.content);
    } catch (error) {
      setSearchHistory([]);
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    getSearchHistory();
  }, []);

  return { searchHistory, isLoading, setSearchHistory };
}

export default useHistory;