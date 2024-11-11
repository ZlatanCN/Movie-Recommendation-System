import express from 'express'
import {
  recommendContentBased
} from '../controllers/recommendationController.js'

const router = express.Router()

// @desc    Get content-based recommendations
// @route   GET /api/recommendation/:id
router.get('/:id', recommendContentBased)

export default router