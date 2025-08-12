from tensorflow import keras
from keras.models import load_model
import cv2
model = load_model('dog_cat_model.h5')
import numpy as np
img_path = "dog.jpg"  # change this to your image path
img = cv2.imread(img_path)  # Read (BGR)
if img is None:
    raise ValueError(f"Image at path '{img_path}' not found.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
img = cv2.resize(img, (150, 150))           # Resize to model input size

# Preprocess
img_array = img.astype("float32") / 255.0   # normalize
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# Predict
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("It's a Dog")
else:
    print("It's a Cat")

print(f"Prediction Probability: {prediction[0][0]:.4f}")