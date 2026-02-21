from tensorflow.keras.models import load_model

# Try loading your eyes model
model = load_model("backend/results/best_model_eyes.h5")
print("Eyes model loaded successfully!")

# Try loading your gills model
model = load_model("backend/results/best_model_gills.h5")
print("Gills model loaded successfully!")
