# Dog🐶 and Cat🐱 Classification
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data)

## 🧠 Project Overview

This project performs binary image classification to distinguish between cats and dogs using a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras.**
The model was trained on the **Kaggle Cats and Dogs dataset** in a **Google Colab** environment.

## 📑 Table of Contents

1. [🧠 Project Overview](#-project-overview)
2. [📂 Project Structure](#-project-structure)
3. [Usage](#usage) 
4. [📓 Model Development (Notebook)](#-model-development-notebook)
    - [Step 1: Dataset Download](#step-1-dataset-download)
    - [Step 2: Data Preprocessing & Augmentation](#step-2-data-preprocessing--augmentation)
        - [📊 Class Distribution](#-class-distribution)
        - [Augmentation](#augmentation)
    - [Step 3: 🔹 CNN Model Architecture](#step-3--cnn-model-architecture)
    - [Step 4: Model Training](#step-4-model-training)
    - [Step 5: Model Evaluation & Results](#step-5-model-evaluation--results)
        - [Results](#results)
        - [Evaluation](#evaluation)
            - [Confusion Matrix and Classification Report](#confusion-matrix-and-classification-report)
5. [🧩Modularization (src)](#-modular-code-src)
6. [Streamlit Application](#-streamlit-application)

---

## 📂 Project Structure
```
1-CNN-Dogs-and-Cats-Classification/
├── 📁 data/
│ └── 📁 sample_images/ 
│
├── 🧠 CNN-Dogs-and-Cats.ipynb # Jupyter Notebook (Google Colab version)
│
├── 📁 src/ 
│ ├── train.py 
│ ├── evaluate.py 
│ ├── predict.py
| ├── prepare_data.py
│ └── model.py 
│
├── 🌐 app/ # Streamlit web application
│ └── app.py 
│
├── 🧩 models/ 
│ └── cat-and-dog.keras 
│
├── 📄 requirements.txt
└── 📘 README.md 
```

> ⚠️ Note: The trained model file (`cat-and-dog.keras`) is not included in this repository due to size constraints. However, you can download the model from my Hugging Face link. Please refer to the [Usage](#usage) section for details.

## Usage
<details> 
    <summary><h3>Step-by-step usage details</h3></summary>

  <br>

**1- To download this project's folder as a `.zip` file, click the link below:**
   [Download](https://download-directory.github.io/?url=https://github.com/havva-nur-ezginci/DeepLearning-Applications/tree/main/1-CNN-Dogs-and-Cats-Classification)

   

**2-Install the dependencies**
   
```bash
pip install -r requirements.txt
```

**3- Prepare Data** : Downloading a dataset using the Kaggle API with the personal kaggle.json file.
```
python src/prepare_data.py
```
⚠️ Note: When using it, do not forget to add your kaggle.json file. 

**4- Train:**
```
python src/train.py --data_dir ../1-CNN-Dogs-and-Cats-Classification/data/training_set/training_set --model_out ../1-CNN-Dogs-and-Cats-Classification/models/cat-and-dog.keras
```

**5- Evaluate:**
```
 python src/evaluate.py --data_dir ../1-CNN-Dogs-and-Cats-Classification/data/test_set/test_set --model_path ../1-CNN-Dogs-and-Cats-Classification/models/cat-and-dog.keras
```

**6- Predict:**
```
python src/predict.py --image ../1-CNN-Dogs-and-Cats-Classification/data/sample_images/dog1.jpg --model ../1-CNN-Dogs-and-Cats-Classification/models/cat-and-dog.keras
```
</details>

#### 📥 Model Download

The trained CNN model used in this project is publicly available on Hugging Face:

➡️ [Download from Hugging Face](https://huggingface.co/havvanur92/dog-cat-cnn/resolve/main/cat-and-dog.keras)

To load the model in your code:
```python
from tensorflow import keras
from huggingface_hub import hf_hub_download

# Download the model file from Hugging Face
model_path = hf_hub_download(repo_id="havvanur92/dog-cat-cnn", filename="cat-and-dog.keras")

# Load the model
model = keras.models.load_model(model_path)
```

---

## 📓 Model Development (Notebook)
The initial development, including data exploration, preprocessing, and model training, was performed in `CNN_Dogs_and_Cats.ipynb`.  

### Step 1: Dataset Download

The dataset for cats and dogs was obtained from Kaggle. First, the Kaggle API key was uploaded to Colab (kaggle.json) to authenticate the account. Then, the dataset was downloaded using the Kaggle CLI:
```sh 
!kaggle datasets download -d tongpython/cat-and-dog
 ```

Samples from the dataset:

<img width="804" height="185" alt="Image" src="https://github.com/user-attachments/assets/bba946c8-406f-4a45-baed-075c5daa1973" />

 ---
### Step 2: Data Preprocessing & Augmentation

#### 📊 Class Distribution

Below are the class distributions for the training and test datasets used in this project:

<p align="center"> 
 <img width="463" height="345" alt="Test Distribution" src="https://github.com/user-attachments/assets/bd50363e-803f-414d-8aa6-79b0bc2764ac" /> 
 <img width="464" height="345" alt="Training Distribution" src="https://github.com/user-attachments/assets/cc4d3dbf-492a-4445-8521-9b22a4c191c1" /> 
</p>

The visualizations show that both datasets are balanced between the two classes (cats 🐱 and dogs 🐶), ensuring fair model training and evaluation.

#### Augmentation

**The dataset was loaded using `ImageDataGenerator` with the `flow_from_directory()` method.**

- Training images were augmented with rotation, horizontal and vertical flips, shear, zoom, and width/height shifts to generate diverse samples, while pixel values were normalized to the [0,1] range. 
- A 20% subset of the training data was reserved for validation using the `validation_split` parameter, ensuring that the training and validation sets do not share any images.
- The training generator (`train_generator`) takes **80%** of the data, while the validation generator (`validation_generator`) uses the remaining **20%**. Test data was loaded separately from a different directory without augmentation to evaluate the model's performance.
- This approach not only enhances model generalization by providing diverse training data but also ensures **efficient memory usage in Google Colab.**

<img width="415" height="277" alt="Image" src="https://github.com/user-attachments/assets/2938fbb1-f96c-40ef-bf1e-f8394c5f898a" />

After preprocessing, the data distribution was as follows:

* Training images: 6404
* Validation images: 1601
* Test images: 2023

**The ImageDataGenerator was used to create diverse and augmented datasets for model training and validation, ensuring efficient memory usage in Google Colab.**

---
### Step 3: 🔹 CNN Model Architecture


<details> 
  <summary><strong>View model</strong></summary> 
  <br>
 <img width="500" height="1000" alt="Image" src="https://github.com/user-attachments/assets/b06d0318-176d-4076-9a41-0f687a3c5cff" />
</details>

The CNN model consists of:

1. **Convolutional layers** with **ReLU** activations to extract features from input images
2. **MaxPooling layers** to reduce spatial dimensions and retain important features
3. **Batch normalization** to improve training stability and convergence
4. **Flatten layer** to convert feature maps into a one-dimensional vector
5. **Dense layers** for classification, including a **dropout layer (35%)** to prevent overfitting
6. **Sigmoid** output layer for binary classification (Cat vs Dog)
---
### Step 4: Model Training 
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
### Step 5: Model Evaluation & Results

#### Results 

The key metrics monitored during training were **Accuracy, Loss, Precision and Recall**.

<p align="center"> 
<img src="https://github.com/user-attachments/assets/353eebbb-ce53-44a3-8c9e-9f32800dd897" alt="Training & Validation Metrics" width="800"/>
 </p>

**Observations :**

**Accuracy**: The model achieved approximately 88% validation accuracy, showing strong performance on unseen data.

**Loss**: Validation loss decreased steadily, indicating effective learning and minimal overfitting.

**Precision & Recall**: Both metrics gradually improved, reaching ~0.85–0.90, which demonstrates a balanced ability to correctly identify both cats and dogs.

#### Evaluation

Model results for test data : 
- Total test image = 2023
- batch_size = 64
=> 31(step)*64 + 39 = 2023
- steps = 32

evaluate => 1-31.(step) : 1984 => 32.(step) : 39

- Test Accuracy: 0.87
- Test Loss: 0.37
- Test Precision: 0.85
- Test Recall: 0.90


##### Confusion Matrix and Classification Report
Below is the confusion matrix and classification report for the test data, which shows the model's performance in predicting two classes.

<p align="center"> 
 <img width="406" height="324" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/5123bfd6-fc19-4d13-8395-399ecfa828d5" />
 <img width="406" height="200" alt="Classification Report" src="https://github.com/user-attachments/assets/248ff276-5326-4856-b846-caf4d9df7f60" />
</p>

📌 Not: The **test set generator was configured with shuffle=False** to ensure that predictions align correctly with the true labels for accurate evaluation.

**=>** After **loading the saved (.keras) model, the prediction results for new images** were examined using the predict_image function.

<p align="center"> 
<img width="250" height="277" alt="Image" src="https://github.com/user-attachments/assets/b44495d1-433a-4a61-8994-aecb1371a304" />
<img width="240" height="273" alt="image" src="https://github.com/user-attachments/assets/b2e68bd7-201b-4a5b-bc7f-328afb3607d6" />
<img width="245" height="277" alt="image" src="https://github.com/user-attachments/assets/02bd90c9-d1bd-4d40-b638-8b9f9cb6d8bc" />
</p>

---

## 🧩 Modular Code (src)
The codebase was refactored into modular scripts under the `src/` directory to ensure scalability and reusability:
- `prepare_data.py` – The dataset is downloaded using the Kaggle API key.
- `train.py` – Train and save the CNN model  
- `evaluate.py` – Evaluate performance on validation/test sets  
- `predict.py` – Make predictions on unseen images  
- `model.py` – Define and compile the CNN architecture
  
---
## 🌐 Streamlit Application
The interactive web interface is built using Streamlit (`app/app.py`).

Run the app locally:
```bash
streamlit run app/app.py
```

<p align="center"> 
<img width="618" height="238" alt="streamlit" src="https://github.com/user-attachments/assets/87e8a90d-4732-4238-bb9d-ecca32e47833" />
<img width="499" height="586" alt="streamlitTestCat" src="https://github.com/user-attachments/assets/49f8cd93-e3e0-485d-b101-9ed0c14d4d2c" />
</p>


---

