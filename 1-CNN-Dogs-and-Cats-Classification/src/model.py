"""
model.py
Defines and compiles the CNN model used for binary (cat/dog) classification.
Framework: TensorFlow / Keras
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2,strides=(2,2)),
        layers.Conv2D(64, (3,3),padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.35),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['acc', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

if __name__ == "__main__":
    m = create_model()
    m.summary()
