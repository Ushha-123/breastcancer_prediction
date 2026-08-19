import os
import cv2
import numpy as np
import tensorflow as tf

from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model only once
model = tf.keras.models.load_model(
    "saved_models/breast_cancer_model.keras"
)

print("✅ Model Loaded Successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read image
    image = cv2.imread(filepath)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    if prediction >= 0.5:
        result = "Malignant"
        confidence = prediction * 100
    else:
        result = "Benign"
        confidence = (1 - prediction) * 100

    return render_template(
        "result.html",
        prediction=result,
        confidence=f"{confidence:.2f}",
        image=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)