import axios from 'axios'
import { PredictionResponse, DetectionResponse } from '@/types'

const API_BASE_URL = 'http://localhost:8001'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for large images
})

export const predictionsAPI = {
  uploadAndPredict: async (file: File): Promise<PredictionResponse> => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post<PredictionResponse>('/predict/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  cameraPredict: async (base64Image: string): Promise<PredictionResponse> => {
    const response = await api.post<PredictionResponse>('/predict/camera', { base64_image: base64Image }, {
      headers: { 'Content-Type': 'application/json' },
    })
    return response.data
  },

  detect: async (file: File): Promise<DetectionResponse> => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post<DetectionResponse>('/detect', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  healthCheck: async () => {
    const response = await api.get('/health')
    return response.data
  },
}
