import { create } from 'zustand';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api/auth';

axios.defaults.withCredentials = true;

const useAuthStore = create((set) => ({
  user: null,
  isAuthenticated: false,
  error: null,
  isLoading: false,
  isCheckingAuth: true,

  signup: async (username, email, password) => {
    set({
      isLoading: true,
      error: null,
    });

    try {
      const response = await axios.post(`${API_URL}/signup`, {
        email,
        username,
        password,
      });

      set({
        user: response.data.user,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error.response.data.message || 'Error in signing up!',
        isLoading: false,
      });

      throw error;
    }
  },

  verifyEmail: async (code) => {
    set({
      isLoading: true,
      error: null,
    });

    try {
      const response = await axios.post(`${API_URL}/verify-email`, { code });

      set({
        user: response.data.user,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error.response.data.message || 'Error in verifying email!',
        isLoading: false,
      });

      throw error;
    }
  },

  login: async (email, password) => {
    set({
      isLoading: true,
      error: null,
    });

    try {
      const response = await axios.post(`${API_URL}/login`, {
        email,
        password,
      });

      set({
        user: response.data.user,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      set({
        error: error.response.data.message || 'Error in logging in!',
        isLoading: false,
      });

      throw error;
    }
  },

  checkAuth: async () => {
    set({
      isCheckingAuth: true,
      error: null,
    });

    try {
      const response = await axios.get(`${API_URL}/check-auth`);

      set({
        user: response.data.user,
        isAuthenticated: true,
        isCheckingAuth: false,
      });
    } catch (error) {
      set({
        error: null,
        isCheckingAuth: false,
        isAuthenticated: false,
      });
    }
  },

  logout: async () => {
    set({
      isLoading: true,
      error: null,
    });

    try {
      await axios.post(`${API_URL}/logout`);

      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      set({
        error: error.response.data.message || 'Error in logging out!',
        isLoading: false,
      });

      throw error;
    }
  },

  forgotPassword: async (email) => {
    set({
      isLoading: true,
      error: null,
    });

    try {
      const response = await axios.post(`${API_URL}/forgot-password`,
        { email });
      set({ isLoading: false, isAuthenticating: false });
    } catch (error) {
      set({
        isLoading: false,
        error: error.response.data.message || 'Error in sending email!',
      });
      throw error;
    }
  },

  resetPassword: async (token, password, confirmPassword) => {
    set({
      isLoading: true,
      error: null,
    });

    try {
      const response = await axios.post(`${API_URL}/reset-password/${token}`,
        { password, confirmPassword });
      set({ isLoading: false, isAuthenticating: false });
    } catch (error) {
      set({
        isLoading: false,
        error: error.response.data.message || 'Error in resetting password!',
      });
      throw error;
    }
  },
}));

export default useAuthStore;