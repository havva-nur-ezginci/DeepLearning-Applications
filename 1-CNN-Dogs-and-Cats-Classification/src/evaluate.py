"""
evaluate.py
Evaluate a saved model on a dataset (test_set).
Usage:
    python src/evaluate.py --data_dir ../1-CNN-Dogs-and-Cats-Classification/data/test_set/test_set --model_path ../models/cat-and-dog.keras
"""

import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../1-CNN-Dogs-and-Cats-Classification/data/test_set/test_set", help="Path to evaluation directory")
    p.add_argument("--model_path", type=str, default="../1-CNN-Dogs-and-Cats-Classification/models/cat-and-dog.keras", help="Path to saved model")
    p.add_argument("--img_size", type=int, nargs=2, default=[224,224], help="Image target size")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return p.parse_args()

def evaluate(args):
    model = tf.keras.models.load_model(os.path.abspath(args.model_path))
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=tuple(args.img_size),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=False
    )

    loss, precision, recall, acc = model.evaluate(test_gen)

    print(f'\nTest Accuracy: %.1f%%' % (100.0 * acc))
    print(f'\nTest Loss: %.1f%%' % (100.0 * loss))
    print(f'\nTest Precision: %.1f%%' % (100.0 * precision))
    print(f'\nTest Recall: %.1f%%' % (100.0 * recall))

    # confusion matrix and classification report
    y_preds = model.predict(test_gen, steps=len(test_gen))

    y_preds_classes = (y_preds.ravel() > 0.5).astype(int)

    y_true = test_gen.classes

    try:
        print("\nClassification report:")
        print(classification_report(y_true, y_preds_classes, target_names=list(test_gen.class_indices.keys())))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_true, y_preds_classes))
    except Exception as e:
        print("sklearn not available, skipping classification report. Install scikit-learn for detailed metrics.")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
