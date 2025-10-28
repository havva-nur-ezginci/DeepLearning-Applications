"""
train.py
Training script that loads images from a directory, trains the model and saves it.
Usage:
    python src/train.py --data_dir ../1-CNN-Dogs-and-Cats-Classification/data/training_set/training_set --model_out ../1-CNN-Dogs-and-Cats-Classification/models/cat-and-dog.keras
"""

import argparse
from model import create_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../1-CNN-Dogs-and-Cats-Classification/data/training_set/training_set", help="Path to training directory (with subfolders per class)")
    p.add_argument("--model_out", type=str, default="../1-CNN-Dogs-and-Cats-Classification/models/cat-and-dog.keras", help="Output path for saved model")
    p.add_argument("--img_size", type=int, nargs=2, default=[224,224], help="Image target size")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    return p.parse_args()

def train(args):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
                                 shear_range=0.2, zoom_range=0.1, horizontal_flip=True, vertical_flip= True)

    train_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=tuple(args.img_size),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='training' 
    )

    val_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=tuple(args.img_size),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = create_model(input_shape=(args.img_size[0], args.img_size[1], 3))

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(filepath=args.model_out,monitor='val_loss',mode='min', save_best_only=True,save_weights_only=False, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
    ]

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=args.epochs,
                        callbacks=callbacks)


    model.save(os.path.abspath(args.model_out))
    return history

if __name__ == "__main__":
    args = parse_args()
    train(args)
