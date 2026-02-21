import { useRef, useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Modal } from '@/components/ui/modal'
import { PredictionDisplay } from '@/components/PredictionDisplay'
import { predictionsAPI } from '@/utils/api'
import { PredictionResponse } from '@/types'

export function CameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [predictions, setPredictions] = useState<PredictionResponse | null>(null)
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [showResultsModal, setShowResultsModal] = useState(false)

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  const startCamera = async () => {
    try {
      setCameraError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
      }
    } catch (error) {
      setCameraError('Unable to access camera. Please check permissions.')
      console.error('Error accessing camera:', error)
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach((track) => track.stop())
      setIsStreaming(false)
    }
  }

  const captureFrame = (): string | null => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d')
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth
        canvasRef.current.height = videoRef.current.videoHeight
        context.drawImage(videoRef.current, 0, 0)
        return canvasRef.current.toDataURL('image/jpeg')
      }
    }
    return null
  }

  const analyzeFrame = async () => {
    try {
      setIsPredicting(true)
      const base64Image = captureFrame()

      if (!base64Image) {
        setCameraError('Failed to capture frame')
        return
      }

      // Remove data URL prefix
      const imageData = base64Image.replace(/^data:image\/\w+;base64,/, '')

      const response = await predictionsAPI.cameraPredict(imageData)
      setPredictions(response)
      if (response.annotated_image) {
        setAnnotatedImage(`data:image/png;base64,${response.annotated_image}`)
      }
      setShowResultsModal(true)
    } catch (error) {
      setCameraError('Error analyzing frame. Please try again.')
      console.error('Error predicting:', error)
    } finally {
      setIsPredicting(false)
    }
  }

  return (
    <div className="container max-w-screen-2xl py-8">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Live Camera Detection</h1>
          <p className="text-muted-foreground mt-2">
            Point your camera at the fish to detect freshness in real-time
          </p>
        </div>

        {/* Camera Section - Centered */}
        <div className="flex justify-center">
          <div className="w-full max-w-2xl">
            <Card>
              <CardHeader>
                <CardTitle>Camera Feed</CardTitle>
                <CardDescription>Point at the fish for detection</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                  {!isStreaming && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
                      <div className="text-center space-y-4">
                        <p className="text-white text-lg">Camera not active</p>
                        <Button onClick={startCamera} className="w-full">
                          Start Camera
                        </Button>
                      </div>
                    </div>
                  )}
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-full object-cover"
                  />
                </div>

                {cameraError && (
                  <div className="p-3 bg-destructive/10 border border-destructive rounded-lg text-sm text-destructive">
                    {cameraError}
                  </div>
                )}

                <div className="grid grid-cols-2 gap-3">
                  <Button
                    onClick={isStreaming ? stopCamera : startCamera}
                    variant={isStreaming ? 'outline' : 'default'}
                    className="w-full"
                  >
                    {isStreaming ? 'Stop Camera' : 'Start Camera'}
                  </Button>
                  <Button
                    onClick={analyzeFrame}
                    disabled={!isStreaming || isPredicting}
                    className="w-full"
                  >
                    {isPredicting ? 'Analyzing...' : 'Analyze'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Results Modal */}
      <Modal
        isOpen={showResultsModal}
        onClose={() => setShowResultsModal(false)}
        title="Fish Freshness Analysis Results"
        size="xl"
      >
        <div className="space-y-6">
          {/* Annotated Image */}
          {annotatedImage && (
            <div className="text-center">
              <h3 className="text-lg font-semibold mb-4">Detected Regions</h3>
              <div className="bg-muted rounded-lg p-4">
                <img
                  src={annotatedImage}
                  alt="Annotated Detection"
                  className="max-w-lg h-auto mx-auto rounded shadow-md"
                />
              </div>
            </div>
          )}

          {/* Analysis Results */}
          {predictions && (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Eye Analysis */}
              {predictions.eye_detected && predictions.eye_prediction ? (
                <PredictionDisplay
                  prediction={predictions.eye_prediction}
                  label="Eye"
                  detected={predictions.eye_detected}
                />
              ) : (
                <PredictionDisplay
                  prediction={null}
                  label="Eye"
                  detected={false}
                />
              )}

              {/* Gill Analysis */}
              {predictions.gill_detected && predictions.gill_prediction ? (
                <PredictionDisplay
                  prediction={predictions.gill_prediction}
                  label="Gill"
                  detected={predictions.gill_detected}
                />
              ) : (
                <PredictionDisplay
                  prediction={null}
                  label="Gill"
                  detected={false}
                />
              )}

              {/* Overall Analysis */}
              {predictions.integrated_prediction && (
                <PredictionDisplay
                  prediction={predictions.integrated_prediction}
                  label="Overall Fish"
                  detected={true}
                />
              )}
            </div>
          )}

          {/* Warning for no detection */}
          {predictions && !predictions.eye_detected && !predictions.gill_detected && (
            <Card className="bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800">
              <CardContent className="pt-6">
                <p className="text-yellow-800 dark:text-yellow-200 text-center">
                  No eyes or gills detected in the image. Please ensure the fish with visible eyes or gills is clearly shown.
                </p>
              </CardContent>
            </Card>
          )}

          {/* Modal Actions */}
          <div className="flex justify-center space-x-4 pt-4 border-t">
            <Button onClick={() => setShowResultsModal(false)} variant="outline">
              Close Results
            </Button>
            <Button onClick={() => setPredictions(null)}>
              Analyze New Frame
            </Button>
          </div>
        </div>
      </Modal>

      {/* Hidden canvas for capturing frames */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}
