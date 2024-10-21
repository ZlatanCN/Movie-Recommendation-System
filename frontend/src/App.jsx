import { Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage.jsx';
import SignUpPage from './pages/SignUpPage.jsx';
import LoginPage from './pages/LoginPage.jsx';

const App = () => {
  return (
    <Routes>
      <Route path={'/'} element={<HomePage/>}/>
      <Route path={'/signup'} element={<SignUpPage/>}/>
      <Route path={'/login'} element={<LoginPage/>}/>
    </Routes>
  )
}

export default App
