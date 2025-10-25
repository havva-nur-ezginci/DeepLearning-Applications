# Dog🐶 and Cat🐱 Classification
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data)

🧠 Deep Learning Model

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs.
The model was trained on the Kaggle Cats vs Dogs dataset using Google Colab.


## Table of Contents




---

## Step 1: Dataset Download

The dataset for cats and dogs was obtained from Kaggle. First, the Kaggle API key was uploaded to Colab (kaggle.json) to authenticate the account. Then, the dataset was downloaded using the Kaggle CLI:
```sh 
!kaggle datasets download -d tongpython/cat-and-dog
 ```

Samples from the dataset:

<img width="804" height="185" alt="Image" src="https://github.com/user-attachments/assets/bba946c8-406f-4a45-baed-075c5daa1973" />

 ---
 ## Step 2: Data Preprocessing & Augmentation

### 📊 Class Distribution

Below are the class distributions for the training and test datasets used in this project:

<p align="center"> 
 <img width="463" height="350" alt="Test Distribution" src="https://github.com/user-attachments/assets/bd50363e-803f-414d-8aa6-79b0bc2764ac" /> 
 <img width="464" height="345" alt="Training Distribution" src="https://github.com/user-attachments/assets/cc4d3dbf-492a-4445-8521-9b22a4c191c1" /> 
</p>

The visualizations show that both datasets are balanced between the two classes (cats 🐱 and dogs 🐶), ensuring fair model training and evaluation.

### Augmentation

Since the dataset contains a large number of images, **batch processing was applied using Keras’ ImageDataGenerator.** This approach also included data augmentation techniques to improve model generalization, such as:

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

**The ImageDataGenerator was used to create diverse and augmented datasets for model training and validation, ensuring efficient memory usage in Google Colab.**

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
## Step 4: Model Training 
The CNN model was trained using the **Keras Sequential API** for binary classification of cats and dogs.

**Training Configuration** :

- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, and Recall
- **Epochs**: 100
- **Validation Split**: 20%
- **Callbacks**:
  - **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
  - **EarlyStopping**: Stops training early if validation loss does not improve, preventing overfitting.
  - **ModelCheckpoint**: Saved the best-performing model based on validation loss as cat-and-dog.keras.

Training was performed on Google Colab’s GPU runtime, which significantly accelerated the process.
The model achieved stable convergence with gradually decreasing validation loss, indicating effective learning and minimal overfitting.

---
## Step 5: Model Evaluation & Results

The key metrics monitored during training were **Accuracy, Loss, Precision and Recall**.

<p align="center"> 
<img src="https://github.com/user-attachments/assets/353eebbb-ce53-44a3-8c9e-9f32800dd897" alt="Training & Validation Metrics" width="800"/>
 </p>

**Observations :**

**Accuracy**: The model achieved approximately 88% validation accuracy, showing strong performance on unseen data.

**Loss**: Validation loss decreased steadily, indicating effective learning and minimal overfitting.

**Precision & Recall**: Both metrics gradually improved, reaching ~0.85–0.90, which demonstrates a balanced ability to correctly identify both cats and dogs.

The smooth convergence of all metrics indicates that the model generalizes well to new images.


---
