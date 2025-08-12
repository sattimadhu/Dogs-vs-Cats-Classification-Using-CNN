import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
# import cv2
import numpy as np

# Data Loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='dogs_cats/train',
    labels='inferred',
    label_mode='binary',        # binary classification
    batch_size=32,
    image_size=(150, 150)
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='dogs_cats/test',
    labels='inferred',
    label_mode='binary',
    batch_size=32,
    image_size=(150, 150)
)

# Normalization

def process(image, label):
    image = tf.cast(image / 255., tf.float32)  # normalize 0-1
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

#model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)


model.save('dog_cat_model.h5')
