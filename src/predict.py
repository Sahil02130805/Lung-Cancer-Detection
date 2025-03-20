import tensorflow as tf
import cv2
import numpy as np

# Load trained model
model = tf.keras.models.load_model("../models/cancer_detector.h5")

# Function to predict if an image is cancerous
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)[0][0]
    return "Cancerous" if prediction > 0.5 else "Non-Cancerous"

# Test the model with an image
image_path = "../test/cancer.jpg"
print(f"Prediction: {predict_image(image_path)}")
