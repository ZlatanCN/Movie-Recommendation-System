import axios from 'axios';
import ENV_VARS from './envVars.js';

const fetchFromTMDB = async (url) => {
  const options = {
    headers: {
      accept: 'application/json',
      Authorization: `Bearer ${ENV_VARS.TMDB_ACCESS_TOKEN}`,
    }
  };

  const response = await axios.get(url, options);

  if (response.status !== 200) {
    throw new Error(`Error in fetchFromTMDB - tmdb: ${response.statusText}`);
  } else {
    return response.data;
  }
}

export default fetchFromTMDB;