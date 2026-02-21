import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export function AboutPage() {
  return (
    <div className="container max-w-4xl py-8">
      <div className="space-y-8">
        <div>
          <h1 className="text-3xl font-bold">About</h1>
          <p className="text-muted-foreground mt-2">
            Learn more about our fish freshness detection system
          </p>
        </div>

        {/* What is this? */}
        <Card>
          <CardHeader>
            <CardTitle>What is Fish Freshness Detection?</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              Fish Freshness Detection is an advanced AI-powered system designed to classify the freshness level
              of milkfish in real-time. Using deep learning models trained on thousands of images, the system
              analyzes specific regions of the fish (eyes and gills) to determine if the fish is fresh, less fresh,
              starting to rot, or rotten.
            </p>
            <p>
              This technology helps fisheries, markets, and restaurants ensure they are providing the highest quality
              products to their customers while minimizing waste from spoiled fish.
            </p>
          </CardContent>
        </Card>

        {/* Technology */}
        <Card>
          <CardHeader>
            <CardTitle>Technology Behind the System</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Deep Learning Models</h4>
              <p className="text-muted-foreground">
                The system uses three specialized ResNet50-based CNN models:
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2 text-muted-foreground">
                <li>Eye Model: Trained on 1000+ cropped eye images</li>
                <li>Gill Model: Trained on 1000+ cropped gill images</li>
                <li>Integrated Model: Trained on full fish images with both eyes and gills</li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Feature Extraction</h4>
              <p className="text-muted-foreground">
                Models extract both visual features through ResNet50 and texture features using Gray-Level Co-occurrence
                Matrix (GLCM) analysis for robust classification.
              </p>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Detection Methods</h4>
              <p className="text-muted-foreground">
                Eyes are detected using Haar Cascade classifiers, while gills are detected using edge detection and
                contour analysis. This ensures accurate region extraction for precise predictions.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Classification Levels */}
        <Card>
          <CardHeader>
            <CardTitle>Classification Levels</CardTitle>
            <CardDescription>Understanding fish freshness categories</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="border-l-4 border-green-500 pl-4">
                <h4 className="font-semibold text-green-700">Fresh</h4>
                <p className="text-sm text-muted-foreground">
                  Clear, bright eyes; red gills; firm texture; excellent smell. Best for immediate consumption.
                </p>
              </div>

              <div className="border-l-4 border-yellow-500 pl-4">
                <h4 className="font-semibold text-yellow-700">Less Fresh</h4>
                <p className="text-sm text-muted-foreground">
                  Slight eye discoloration; fading gill color; still acceptable. Use within 1-2 days.
                </p>
              </div>

              <div className="border-l-4 border-orange-500 pl-4">
                <h4 className="font-semibold text-orange-700">Starting to Rot</h4>
                <p className="text-sm text-muted-foreground">
                  Dull eyes; brown gills; soft texture; unpleasant smell. Use with caution, cook immediately.
                </p>
              </div>

              <div className="border-l-4 border-red-500 pl-4">
                <h4 className="font-semibold text-red-700">Rotten</h4>
                <p className="text-sm text-muted-foreground">
                  Sunken eyes; gray gills; mushy texture; strong foul odor. Do NOT consume.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* How to Use */}
        <Card>
          <CardHeader>
            <CardTitle>How to Use</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Option 1: Live Camera Detection</h4>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                <li>Click "Live Camera Detection" from the homepage</li>
                <li>Allow camera access when prompted</li>
                <li>Position the fish in front of the camera</li>
                <li>Click "Analyze" to get instant freshness predictions</li>
              </ol>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Option 2: Image Upload</h4>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                <li>Click "Upload Image" from the homepage</li>
                <li>Drag and drop an image or click to select</li>
                <li>Click "Analyze Image"</li>
                <li>View detailed predictions with confidence scores</li>
              </ol>
            </div>
          </CardContent>
        </Card>

        {/* Best Practices */}
        <Card>
          <CardHeader>
            <CardTitle>Best Practices for Accurate Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-muted-foreground">
              <li>✓ Ensure good lighting conditions</li>
              <li>✓ Position the fish clearly with visible eyes and/or gills</li>
              <li>✓ Avoid shadows and glare on the fish</li>
              <li>✓ Use images or camera frames with at least 224x224 pixel resolution</li>
              <li>✓ For best results, position the head of the fish facing the camera</li>
              <li>✓ Ensure the fish is not wrapped or covered</li>
            </ul>
          </CardContent>
        </Card>

        {/* Disclaimer */}
        <Card className="bg-muted/50 border-muted">
          <CardHeader>
            <CardTitle className="text-base">Disclaimer</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            This AI system is designed as a tool to assist in fish quality assessment. Professional judgment from
            trained personnel should always be used in conjunction with this system for final decisions. The system
            should not be the sole determinant of fish safety for consumption.
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
