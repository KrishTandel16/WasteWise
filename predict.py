import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.utils import load_img, img_to_array

# Load model
model = tf.keras.models.load_model(
    "models/wastewise_best.keras"
)

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Change this to the image you want to test
IMAGE_PATH = "image.jpeg"

# Load image
image = load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

image_array = img_to_array(image)

# Normalize
image_array = image_array / 255.0

# Add batch dimension
image_array = np.expand_dims(
    image_array,
    axis=0
)

# Predict
predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print()
print("==============================")
print("       WASTEWISE PREDICTION")
print("==============================")
print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
print("==============================")