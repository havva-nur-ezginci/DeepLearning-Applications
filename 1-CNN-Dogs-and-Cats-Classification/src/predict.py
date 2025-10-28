"""
predict.py
Load a saved model and run prediction on a single image.
Usage:
    python src/predict.py --image ../1-CNN-Dogs-and-Cats-Classification/data/sample_images/dog1.jpg
or
    python src/predict.py --image ../1-CNN-Dogs-and-Cats-Classification/data/sample_images/dog1.jpg --model ../1-CNN-Dogs-and-Cats-Classification/models/cat-and-dog.keras
"""

import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--model", type=str, default="../1-CNN-Dogs-and-Cats-Classification/models/cat-and-dog.keras", help="Path to saved model")
    p.add_argument("--img_size", type=int, nargs=2, default=[224,224], help="Image target size")
    return p.parse_args()

def predict_single(args):
    model = tf.keras.models.load_model(os.path.abspath(args.model))
    img = image.load_img(args.image, target_size=tuple(args.img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array , axis=0)
    p = model.predict(img_array )[0][0]
    label = 'dog' if p > 0.5 else 'cat'
    print(f"Prediction: {label} (score={p:.4f})")
    return label, float(p)

if __name__ == "__main__":
    args = parse_args()
    predict_single(args)
