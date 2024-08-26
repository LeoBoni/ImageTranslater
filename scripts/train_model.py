import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

def load_dataset():
    X, y = [], []
    labels = {letter: idx for idx, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}

    for letter in os.listdir('dataset'):
        folder_path = f'dataset/{letter}'
        for img_file in os.listdir(folder_path):
            if img_file.endswith('.npy'):
                img_path = os.path.join(folder_path, img_file)
                X.append(np.load(img_path))
                y.append(labels[letter])

    return np.array(X), tf.keras.utils.to_categorical(np.array(y), num_classes=26)

X, y = load_dataset()

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, validation_split=0.2)

model.save('models/libras_alphabet_model.h5')
