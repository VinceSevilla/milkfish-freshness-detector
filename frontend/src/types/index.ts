export interface Prediction {
  class: string
  confidence: number
  probabilities: Record<string, number>
}

export interface PredictionResponse {
  eye_detected: boolean
  gill_detected: boolean
  eye_prediction: Prediction | null
  gill_prediction: Prediction | null
  integrated_prediction: Prediction | null
  annotated_image: string | null
  original_image_base64?: string
}

export interface DetectionResponse {
  eye_detected: boolean
  gill_detected: boolean
  eye_bbox: [number, number, number, number] | null
  gill_bbox: [number, number, number, number] | null
  original_image: string
  annotated_image: string
  eye_roi: string | null
  gill_roi: string | null
}
