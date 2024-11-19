import NavBar from '../components/NavBar.jsx'
import { motion } from 'framer-motion'
import { useState } from 'react'
import { LeftOutlined, LoadingOutlined, MailOutlined } from '@ant-design/icons'
import { Input, message } from 'antd'
import useAuthStore from '../store/authStore.js'
import { Link } from 'react-router-dom'

const ForgotPasswordPage = () => {
  const [messageApi, contextHolder] = message.useMessage()
  const [hasSubmitted, setHasSubmitted] = useState(false)
  const [email, setEmail] = useState('')
  const { isLoading, forgotPassword } = useAuthStore()

  const handleSubmit = async (e) => {
    e.preventDefault()
    await forgotPassword(email).
      then(() => setHasSubmitted(true)).
      catch((error) => {
        messageApi.error({
          content: error.response.data.message || 'Error in sending email!',
          className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
        })
        console.log(error.response.data)
      })
  }

  return (
    <div className={'h-screen w-full hero-bg bg-center'}>
      {contextHolder}

      {/* NavBar */}
      <NavBar type={'auth'}/>

      <section className={'flex justify-center items-center mt-20 mx-3 '}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className={'max-w-md w-full bg-gray-800 bg-opacity-50 backdrop-filter backdrop-blur-xl rounded-2xl shadow-xl overflow-hidden'}
        >
          <div className={'p-8'}>
            {/*Forgot Password Title*/}
            <h2
              className={'text-3xl font-bold text-center bg-gradient-to-r from-red-600 to-red-700 text-transparent bg-clip-text mb-6'}>
              Forgot Password
            </h2>

            {/*Forgot Password Form*/}
            {!hasSubmitted ? (
              <form onSubmit={handleSubmit} className={'flex flex-col gap-6 '}>
                <p className={'text-gray-300 text-center'}>
                  Enter your email address and we&#39;ll send you a link to
                  reset
                  your password.
                </p>
                <Input
                  prefix={<MailOutlined className={'text-red-600 pr-1'}/>}
                  placeholder={'Email Address'}
                  type={'email'}
                  value={email}
                  size={'large'}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className={'bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 text-white'}
                />
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  type={'submit'}
                  className={'w-full py-3 px-4 bg-gradient-to-r from-red-600 to-red-700 text-white font-bold rounded-lg shadow-lg hover:from-red-700 hover:to-red-800 focus:outline-none focus:ring-1 focus:ring-red-600 focus:ring-offset-1 focus:ring-offset-gray-900'}
                >
                  {isLoading ? <LoadingOutlined/> : 'Send Reset Link'}
                </motion.button>
              </form>
            ) : (
              <section className={'text-center flex flex-col gap-6'}>
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                >
                  <MailOutlined className={'text-red-600 text-3xl'}/>
                </motion.div>
                <p className={'text-gray-300'}>
                  If an account exists for {email}, you will receive a password
                  reset link shortly.
                </p>
              </section>
            )}
          </div>

          <footer
            className={'px-8 py-4 bg-gray-800 bg-opacity-50 flex justify-center'}>
            <Link to={'/login'}
                  className={'text-sm text-red-500 hover:underline flex gap-1'}>
              <LeftOutlined/>
              <span>Back to Login</span>
            </Link>
          </footer>
        </motion.div>
      </section>
    </div>
  )
}

export default ForgotPasswordPage
