import express from 'express'
import {
  recommendContentBased,
  recommendCollaborative
} from '../controllers/recommendationController.js'

const router = express.Router()

// @desc    Get content-based recommendations
// @route   GET /api/recommendation/content/:id
router.get('/content/:id', recommendContentBased)

// @desc    Get collaborative filtering recommendations
// @route   GET /api/recommendation/collaborative
router.get('/collaborative', recommendCollaborative)

export default router