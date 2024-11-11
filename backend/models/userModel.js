import mongoose from 'mongoose';

const userSchema = mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
  },
  password: {
    type: String,
    required: true,
  },
  image: {
    type: String,
    default: ''
  },
  searchHistory: {
    type: Array,
    default: []
  },
  ratedMovies: {
    type: Array,
    default: []
  },
  lastLogin: {
    type: Date,
    default: Date.now
  },
  isVerified: {
    type: Boolean,
    default: false
  },
  resetPasswordToken: String,
  resetPasswordExpiredAt: Date,
  verificationToken: String,
  verificationTokenExpiredAt: Date,
}, {timestamps: true});

const User = mongoose.model('User', userSchema);

export default User;