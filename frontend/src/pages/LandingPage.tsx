import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export function LandingPage() {
  const navigate = useNavigate()

  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 md:py-32 px-4">
        <div className="container max-w-4xl mx-auto text-center space-y-8">
          <div className="space-y-4">
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
              Fish Freshness Detection
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground">
              Advanced AI-powered system for real-time fish quality assessment using deep learning
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button onClick={() => navigate('/camera')} size="lg" className="text-lg h-12 px-8">
              Live Camera Detection
            </Button>
            <Button
              onClick={() => navigate('/upload')}
              size="lg"
              variant="outline"
              className="text-lg h-12 px-8"
            >
              Upload Image
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-muted/50">
        <div className="container max-w-5xl">
          <h2 className="text-3xl font-bold mb-12 text-center">Key Features</h2>

          <div className="grid md:grid-cols-3 gap-8">
            <Card>
              <CardHeader className="text-center">
                <div className="mb-2 flex justify-center">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-black dark:text-white">
                    <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
                    <circle cx="12" cy="13" r="3"/>
                  </svg>
                </div>
                <CardTitle>Live Camera Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Real-time fish freshness detection using your webcam. Automatic eye and gill detection with instant predictions.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <div className="mb-2 flex justify-center">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-black dark:text-white">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                </div>
                <CardTitle>Image Upload</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Upload images from your device. Supports drag-and-drop functionality for ease of use.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <div className="mb-2 flex justify-center">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-black dark:text-white">
                    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
                    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
                  </svg>
                </div>
                <CardTitle>Deep Learning Models</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  ResNet50-based CNN with GLCM texture features for accurate classification into 4 freshness levels.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <div className="mb-2 flex justify-center">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-black dark:text-white">
                    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                    <circle cx="12" cy="12" r="3"/>
                  </svg>
                </div>
                <CardTitle>Eye Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Automatic eye region detection and extraction. Predicts freshness based on eye appearance patterns.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <div className="mb-2 flex justify-center">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-black dark:text-white">
                    <path d="M6.5 12c.94-3.46 4.94-6 8.5-6 3.56 0 6.06 2.54 7 6-1 3.5-3.5 6-7 6s-6.06-2.54-7-6Z"/>
                    <path d="M18 12v.5"/>
                    <path d="M16 17.93a9.77 9.77 0 0 1-2 .07c-3.56 0-6.06-2.54-7-6 .94-3.46 4.94-6 8.5-6 3.56 0 6.06 2.54 7 6-.94 3.46-3.44 6-6.5 6Z"/>
                    <path d="M2 12h7"/>
                  </svg>
                </div>
                <CardTitle>Gill Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Intelligent gill region detection. Analyzes color and texture for accurate freshness assessment.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center">
                <div className="mb-2 flex justify-center">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-black dark:text-white">
                    <line x1="18" y1="20" x2="18" y2="10"/>
                    <line x1="12" y1="20" x2="12" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="14"/>
                  </svg>
                </div>
                <CardTitle>Detailed Results</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Get confidence scores, probability distributions, and annotated images with detected regions.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Freshness Levels Section */}
      <section className="py-16">
        <div className="container max-w-4xl">
          <h2 className="text-3xl font-bold mb-12 text-center">Freshness Classification</h2>

          <div className="space-y-4">
            <div className="p-4 rounded-lg border-2 bg-green-50 border-green-300">
              <h3 className="font-bold text-lg text-green-900">Fresh</h3>
              <p className="text-green-700">Excellent condition, bright eyes, red gills - ideal for consumption</p>
            </div>

            <div className="p-4 rounded-lg border-2 bg-yellow-50 border-yellow-300">
              <h3 className="font-bold text-lg text-yellow-900">Less Fresh</h3>
              <p className="text-yellow-700">Slight discoloration, mild color changes - still acceptable</p>
            </div>

            <div className="p-4 rounded-lg border-2 bg-orange-50 border-orange-300">
              <h3 className="font-bold text-lg text-orange-900">Starting to Rot</h3>
              <p className="text-orange-700">Noticeable discoloration, unpleasant odor - use with caution</p>
            </div>

            <div className="p-4 rounded-lg border-2 bg-red-50 border-red-300">
              <h3 className="font-bold text-lg text-red-900">Rotten</h3>
              <p className="text-red-700">Severe decomposition, strong odor - not suitable for consumption</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
