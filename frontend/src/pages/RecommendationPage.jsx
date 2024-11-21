import NavBar from '../components/NavBar.jsx';
import { motion } from 'framer-motion';
import MovieRecommendationCard from '../components/MovieRecommendationCard.jsx';

const RecommendationPage = () => {

  return (
    <div className={'bg-black text-white min-h-screen relative'}>
      {/* Navigation */}
      <NavBar/>

      {/* Main Content */}
      <main className={'max-w-6xl mx-auto px-4 py-8'}>
        <h1 className={'text-3xl font-bold mb-8'}>
          Your Recommendations
        </h1>

        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className={'grid grid-cols-1 min-h-[70vh]'}
        >
          <MovieRecommendationCard/>
        </motion.div>
      </main>
    </div>
  );
};

export default RecommendationPage;