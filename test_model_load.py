from tensorflow.keras.models import load_model

try:
    eye_model = load_model("backend/results/best_model_eyes.h5", compile=False)
    print("Eye model loaded successfully!")
except Exception as e:
    print("Failed to load eye model:", e)

try:
    gill_model = load_model("backend/results/best_model_gills.h5", compile=False)
    print("Gill model loaded successfully!")
except Exception as e:
    print("Failed to load gill model:", e)
