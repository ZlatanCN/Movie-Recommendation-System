import { Input, message } from 'antd'
import { motion } from 'framer-motion'
import { useState } from 'react'
import useAuthStore from '../store/authStore.js'
import { useNavigate } from 'react-router-dom'

const EmailVerificationPage = () => {
  const [verificationCode, setVerificationCode] = useState('')
  const [messageApi, contextHolder] = message.useMessage()
  const { verifyEmail } = useAuthStore()
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()

    try {
      await verifyEmail(verificationCode).then(() => {
        messageApi.success({
          content: 'Welcome to the community!',
          className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
          duration: 2,
        })
        setTimeout(() => navigate('/'), 2000)
      })
    } catch (error) {
      messageApi.error({
        content: 'Invalid Code',
        className: 'text-gray-300 font-bold font-mono mt-20 text-[16px]',
      })
      console.error(error)
    }
  }

  return (
    <div className={'h-screen w-full hero-bg bg-center'}>
      {contextHolder}
      <section className={'flex justify-center items-center h-full'}>
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className={'bg-gray-800 bg-opacity-50 backdrop-filter backdrop-blur-xl rounded-2xl shadow-xl p-8 w-full max-w-md flex flex-col gap-6'}
        >
          {/* Verification Title */}
          <h2
            className={'text-3xl font-bold text-center bg-gradient-to-r from-red-600 to-red-700 text-transparent bg-clip-text'}>
            Verify Your Email
          </h2>

          {/* Verification Instructions */}
          <p className={'text-center text-gray-300'}>
            Enter the 6-digit code sent to your email address.
          </p>

          {/* Verification Form */}
          <form className={'space-y-6'} onSubmit={handleSubmit}>
            {/* Verification Code */}
            <Input.OTP
              length={6}
              onChange={(value) => setVerificationCode(value)}
            />

            {/* Submit Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type={'submit'}
              className={'w-full py-3 px-4 bg-gradient-to-r from-red-600 to-red-700 text-white font-bold rounded-lg shadow-lg hover:from-red-700 hover:to-red-800 focus:outline-none focus:ring-1 focus:ring-red-600 focus:ring-offset-1 focus:ring-offset-gray-900'}
            >
              Submit
            </motion.button>
          </form>
        </motion.div>
      </section>
    </div>
  )
}

export default EmailVerificationPage
