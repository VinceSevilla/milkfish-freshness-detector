import { Prediction } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface PredictionDisplayProps {
  prediction: Prediction | null
  label: string
  detected: boolean
}

function getColorForClass(freshness: string): string {
  switch (freshness) {
    case 'Fresh':
      return 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-100 border-green-300 dark:border-green-700'
    case 'Less Fresh':
      return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-100 border-yellow-300 dark:border-yellow-700'
    case 'Starting to Rot':
      return 'bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-100 border-orange-300 dark:border-orange-700'
    case 'Rotten':
      return 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-100 border-red-300 dark:border-red-700'
    default:
      return 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-100 border-gray-300 dark:border-gray-600'
  }
}

export function PredictionDisplay({ prediction, label, detected }: PredictionDisplayProps) {
  if (!detected) {
    return (
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-base">{label}</CardTitle>
          <CardDescription>Not detected in image</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No {label.toLowerCase()} region was detected. Please upload or capture an image with a clear {label.toLowerCase()} region.
          </p>
        </CardContent>
      </Card>
    )
  }

  if (!prediction) {
    return (
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-base">{label}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Processing...</p>
        </CardContent>
      </Card>
    )
  }

  const colorClass = getColorForClass(prediction.class)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{label}</CardTitle>
        <CardDescription>Freshness Classification</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className={`p-4 rounded-lg border-2 ${colorClass} text-center`}>
          <p className="font-bold text-lg">{prediction.class}</p>
          <p className="text-sm">Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
        </div>

        <div className="space-y-2">
          <p className="text-sm font-semibold">Probability Distribution:</p>
          {Object.entries(prediction.probabilities).map(([cls, prob]) => (
            <div key={cls} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span>{cls}</span>
                <span>{(prob * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all ${
                    cls === prediction.class 
                      ? 'bg-blue-500 dark:bg-blue-400' 
                      : 'bg-gray-300 dark:bg-gray-600'
                  }`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
