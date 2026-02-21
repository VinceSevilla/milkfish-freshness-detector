# Fish Freshness Detection

A comprehensive AI-powered system for real-time fish freshness classification using deep learning.

## Features

- **Live Camera Detection**: Real-time fish freshness detection using webcam
- **Image Upload**: Batch image analysis with drag-and-drop support
- **Eye Detection**: Automatic eye region extraction and analysis
- **Gill Detection**: Automatic gill region extraction and analysis
- **Multi-Model Predictions**: Eye, gill, and integrated fish predictions
- **Annotated Results**: Visual feedback with detected regions highlighted
- **Confidence Scoring**: Detailed probability distributions for each prediction

## Project Structure

```
Fish_Freshness_Detection/
├── backend/                 # FastAPI backend server
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── detection/      # Eye/Gill detection module
│   │   └── models_service/ # Model loading and inference
│   └── requirements.txt     # Python dependencies
├── frontend/               # React web application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── utils/          # Utility functions
│   │   ├── types/          # TypeScript types
│   │   └── App.tsx         # Main app component
│   └── package.json        # Node dependencies
├── models/                 # Model directory (for trained models)
├── results/                # Trained models
│   ├── best_model_eyes.h5
│   ├── best_model_gills.h5
│   └── best_model_eyes_and_gills.h5
└── data/                   # Training/testing data
```

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the backend server:
```bash
python -m uvicorn app.main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

### `/health` (GET)
Health check endpoint

### `/predict/upload` (POST)
Upload an image and get complete predictions

**Request**: FormData with `file` field
**Response**:
```json
{
  "eye_detected": boolean,
  "gill_detected": boolean,
  "eye_prediction": {
    "class": "Fresh|Less Fresh|Starting to Rot|Rotten",
    "confidence": number,
    "probabilities": {}
  },
  "gill_prediction": { ... },
  "integrated_prediction": { ... },
  "annotated_image": "base64_string"
}
```

### `/predict/camera` (POST)
Analyze a camera frame (base64 encoded)

**Request**: JSON with `base64_image` field
**Response**: Same as `/predict/upload`

### `/detect` (POST)
Detect regions without predictions

**Request**: FormData with `file` field
**Response**:
```json
{
  "eye_detected": boolean,
  "gill_detected": boolean,
  "eye_bbox": [x, y, w, h],
  "gill_bbox": [x, y, w, h],
  "annotated_image": "base64_string",
  "eye_roi": "base64_string",
  "gill_roi": "base64_string"
}
```

## Classification Levels

- **Fresh**: Clear bright eyes, red gills, excellent condition
- **Less Fresh**: Slight discoloration, mild color changes
- **Starting to Rot**: Noticeable discoloration, unpleasant odor
- **Rotten**: Severe decomposition, strong foul odor

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Image Processing**: NumPy, Pillow, scikit-image

### Frontend
- **Framework**: React 18
- **Language**: TypeScript
- **UI Components**: shadcn/ui
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **HTTP Client**: Axios
- **Router**: React Router v6

## Models Information

### Eye Model (best_model_eyes.h5)
- Trained on 1000+ cropped eye images
- ResNet50 backbone with GLCM texture features
- 4-class classification

### Gill Model (best_model_gills.h5)
- Trained on 1000+ cropped gill images
- ResNet50 backbone with GLCM texture features
- 4-class classification

### Integrated Model (best_model_eyes_and_gills.h5)
- Trained on complete fish images
- ResNet50 backbone for full-image analysis
- 4-class classification

## Best Practices

- Ensure good lighting conditions
- Position fish clearly with visible eyes and/or gills
- Avoid shadows and glare
- Use images with at least 224x224 pixel resolution
- Position fish head facing the camera for best results
- Keep fish uncovered and fully visible

## Troubleshooting

### Camera Not Working
- Check browser permissions for camera access
- Ensure camera is not in use by another application
- Try a different browser

### API Connection Failed
- Ensure backend is running on port 8000
- Check firewall settings
- Verify CORS is enabled

### Low Detection Accuracy
- Improve lighting conditions
- Ensure fish is clearly visible
- Check if fish has visible eyes or gills
- Try adjusting the camera angle

## Future Enhancements

- Real-time video analysis with continuous detection
- Multiple fish detection in a single frame
- Model export and optimization for edge devices
- Mobile application support
- Database integration for historical tracking
- Advanced visualization tools

## License

This project is part of the Fish Freshness Detection Research System.

## Support

For issues, questions, or contributions, please contact the development team.
