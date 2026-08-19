import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "saved_models/breast_cancer_model.keras"

# Load Model
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model Loaded Successfully!")

# Image Path
IMAGE_PATH = input("Enter image path: ")

# Read Image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Image not found!")
    exit()

# Convert BGR → RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize
image = cv2.resize(image, (224, 224))

# Normalize
image = image / 255.0

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Prediction
prediction = model.predict(image)[0][0]

print("\nPrediction Score:", prediction)

if prediction >= 0.5:
    print("🔴 Prediction: Malignant")
    confidence = prediction * 100
else:
    print("🟢 Prediction: Benign")
    confidence = (1 - prediction) * 100

print(f"Confidence: {confidence:.2f}%")