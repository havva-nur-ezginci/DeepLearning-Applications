# Dog🐶 and Cat🐱 Classification
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data)

🧠 Deep Learning Model

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs.
The model was trained on the Kaggle Cats vs Dogs dataset using Google Colab.

---

## Step 1: Dataset Download

The dataset for cats and dogs was obtained from Kaggle. First, the Kaggle API key was uploaded to Colab (kaggle.json) to authenticate the account. Then, the dataset was downloaded using the Kaggle CLI:
```sh 
!kaggle datasets download -d tongpython/cat-and-dog
 ```
 ---
 ## Step 2: Data Preprocessing & Augmentation

Since the dataset contains a large number of images, batch processing was applied using Keras’ ImageDataGenerator. This approach also included data augmentation techniques to improve model generalization, such as:

* Rescaling pixel values
* Random rotations
* Horizontal and vertical flips
* Shearing and zooming
* Width and height shifts

The dataset was split into training (80%) and validation (20%) subsets. A separate test set was also prepared.

<img width="415" height="277" alt="Image" src="https://github.com/user-attachments/assets/2938fbb1-f96c-40ef-bf1e-f8394c5f898a" />

After preprocessing, the data distribution was as follows:

* Training images: 6404
* Validation images: 1601
* Test images: 2023

This setup ensured efficient memory usage on Colab while providing diverse and augmented data for model training.

---
## Step 3: 🔹 CNN Model Architecture

<img width="7507" height="123" alt="Image" src="https://github.com/user-attachments/assets/fe3f8323-1ffd-431d-a39c-19c908d7ad49" />

The CNN model consists of:

1. **Convolutional layers** with **ReLU** activations to extract features from input images
2. **MaxPooling layers** to reduce spatial dimensions and retain important features
3. **Batch normalization** to improve training stability and convergence
4. **Flatten layer** to convert feature maps into a one-dimensional vector
5. **Dense layers** for classification, including a **dropout layer (35%)** to prevent overfitting
6. **Sigmoid** output layer for binary classification (Cat vs Dog)
---
## Step 4: Model Training Setup

Before training, the model was compiled with the following settings:

- **Loss function**: Binary Crossentropy (for binary classification)

- **Optimizer**: Adam (adaptive learning rate)

- **Metrics**: Accuracy, Precision, and Recall

To improve **training efficiency** and **prevent overfitting**, callbacks were applied:

- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus

- **EarlyStopping**: Stops training early if validation loss does not improve

- **ModelCheckpoint**: Saves the best model based on validation loss

These configurations ensured that the model trained optimally while avoiding overfitting and saved the best-performing weights for later use.

---
