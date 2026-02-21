# Fish Freshness Detection - Backend

FastAPI backend for real-time fish freshness detection using pre-trained deep learning models.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python -m uvicorn app.main:app --reload --port 8000
```

3. Check health:
```bash
curl http://localhost:8000/health
```

## Endpoints

- GET `/health` - Health check
- POST `/predict/upload` - Upload image for prediction
- POST `/predict/camera` - Analyze camera frame
- POST `/detect` - Detect regions only

## Models

Place the following model files in the `results/` directory:
- best_model_eyes.h5
- best_model_gills.h5
- best_model_eyes_and_gills.h5
