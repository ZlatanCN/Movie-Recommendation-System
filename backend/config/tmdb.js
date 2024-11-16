import axios from 'axios';
import ENV_VARS from './envVars.js';

const fetchFromTMDB = async (url) => {
  const options = {
    headers: {
      accept: 'application/json',
      Authorization: `Bearer ${ENV_VARS.TMDB_ACCESS_TOKEN}`,
    }
  };

  try {
    const response = await axios.get(url, options);

    if (response.status !== 200) {
      console.log(`Error in fetchFromTMDB - tmdb: ${response.statusText}`);
      return null;
    } else {
      return response.data;
    }
  } catch (error) {
    console.log(`Error in fetchFromTMDB - tmdb: ${error.message}`);
    return null;
  }
}

export default fetchFromTMDB;