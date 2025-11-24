"# DeepLearning-Applications" 
---
## DL Resources

https://keras.io/api/

https://poloclub.github.io/cnn-explainer/ 

https://netron.app/

---

## Table of Contents
- [CNN](#cnn)
  - [1-CNN-Dogs-and-Cats-Classification](#dog---cat-classification-with-cnn)
- [RNN](#rnn)
  - [1-RNN-Household-Electric-Power-Consumption](#-household-electric-power-consumption-forecasting-with-rnn)
  - [2-RNN-News-Topic-Classification](#-news-topic-classification-with-rnn)

---
# CNN


---

## Dog🐶 - Cat🐱 Classification with CNN
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data)   

> **For more detailed information, please refer to the**
> [➡️ **Read the full README**](https://github.com/havva-nur-ezginci/DeepLearning-Applications/tree/main/1-CNN-Dogs-and-Cats-Classification)

---

### 🧠 Overview

This project performs **binary image classification** (Cats vs Dogs) using a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras**.
It was trained on the **Kaggle** Cats and Dogs dataset within a **Google Colab** environment.

**Model performance on the test set:**
> 🧩 Accuracy: 0.87 | Precision: 0.85 | Recall: 0.90

Below is the **confusion matrix and classification report** of the model’s test results:
<p align="center"> 
 <img width="406" height="324" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/5123bfd6-fc19-4d13-8395-399ecfa828d5" />
 <img width="406" height="200" alt="Classification Report" src="https://github.com/user-attachments/assets/248ff276-5326-4856-b846-caf4d9df7f60" />
</p>

- To improve generalization, **ImageDataGenerator and data augmentation** techniques were applied.
- The project was carried out in the **Google Colab** environment.
- The notebook was later refactored into a **modular Python structure**, and a **Streamlit web app** was developed for user-friendly interaction with the trained model.

### ⚙️ Tech Stack

- **Python, TensorFlow, Keras, NumPy, Matplotlib**
- **Google Colab, Streamlit**
- **Hugging Face**

<details> 
  <summary><h3>📂 Project Structure</h3></summary> 
  <pre><code>
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
  </code></pre>
</details>

> ⚠️ Note: The trained model file (`cat-and-dog.keras`) is not included in this repository due to size constraints. However, you can download the model from my Hugging Face link. For details, see the project's **README → Usage** section.
---

# RNN

---

## ⚡ Household Electric Power Consumption Forecasting with RNN

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

> **For more detailed information, please refer to the**
> [➡️ **Read the full README**](https://github.com/havva-nur-ezginci/DeepLearning-Applications/tree/main/1-RNN-Household-Electric-Power-Consumption)

---

**Project Goal**: Forecast household electricity consumption using a **Recurrent Neural Network (RNN)**.

### 📌 Key Highlights
- **Dataset**: Minute-level electricity consumption from Dec 2006 – Nov 2010 (~2 million records).
- **Target**: **`Global_active_power (kW)`**
- **Time Resolution**: Resampled to **hourly averages** for easier analysis and reduced missing data.

### 🧹 Data Processing & Features
- **Datetime parsing and indexing**
- **Missing value handling**: Interpolation for short gaps (≤3h), removal for longer gaps.
- **Feature engineering**: Added time-based features (`hour`,`is_weekend`,`hour_sin`,`hour_cos`) to capture daily and weekly patterns.
- **Feature selection**: Dropped low-correlation columns (`dayofweek`,`month`).
- **Lag analysis**: Determined 24-hour lookback window for RNN sequences.
- **24-hour lookback window predicting the next hour**
- Data split: **70% train / 15% validation / 15% test**
- Feature scaling using **StandardScaler** (fit on training set only)

 ### 🏗️ Model Overview
- **Architecture**: Two-layer **SimpleRNN** with **ReLU** activations and **Dropout** regularization, followed by Dense output.
- **Loss / Metrics**: MSE, MAE
- **Optimizer**: Adam with **gradient clipping (clipnorm=2.0)**
- **Training**: Early stopping based on validation loss, model checkpointing.

### 📊 Results

**Predictions** were generated for the training, validation, and test sets using the **best saved model**. The results were converted back to the **original scale** and evaluated using **MSE, RMSE, and MAE** metrics.

  
| Dataset     | MSE   | RMSE  | MAE   |
|------------|-------|-------|-------|
| Train      | 0.264 | 0.514 | 0.351 |
| Validation | 0.281 | 0.530 | 0.368 |
| Test       | 0.222 | 0.471 | 0.330 |

<img width="85%" height="420" alt="Global Active Power Prediction Plot" src="https://github.com/user-attachments/assets/86045e5a-5f32-432c-992e-eecb8e6c6416" />


### Dependencies
- Python 3.x, TensorFlow 2.x
- Libraries: `numpy`,`pandas`,`matplotlib`,`seaborn`,`scikit-learn`,`tensorflow/keras`
- Developed on **Google Colab**

```sh
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

 ----
 
## 📰 News Topic Classification with RNN
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data)


> **For more detailed information, please refer to the**
> [➡️ **Read the full README**](https://github.com/havva-nur-ezginci/DeepLearning-Applications/tree/main/2-RNN-News-Topic-Classification)

---

**Project Goal**: Classify news articles from the AG News dataset into **four categories (World, Sports, Business, Science/Technology)** using a **Recurrent Neural Network (RNN)**.

### 📌 Key Highlights

- **Dataset**: 127,600 samples (120k training + 7.6k test) with title and description.

- **Target**: News category label (4 classes).

- **Preprocessing & Features**:

   - Merged title and description into a single text input
   - Tokenized with 10,000-word vocabulary + `<OOV>` token
   - Converted to integer sequences
   - Padded/truncated sequences to `maxlen = 100`

### 🏗️ Model Overview

- **Architecture**: Embedding → SimpleRNN → Dropout → Dense (Softmax)

- **Hyperparameters**: Embedding dimension = 128, RNN units = 160, Dropout = 0.3, Clipnorm = 1.0

- **Hyperparameter Tuning**: Random Search via Keras Tuner

- **Training**: Adam optimizer, categorical crossentropy, gradient clipping, early stopping

### 📊 Results

 The final model was evaluated on the test set : 

- **Accuracy**: 0.8915
- **Loss**: 0.3542
- **AUC**: 0.9756

- **Evaluation  Visualizations**: ROC curves, Classification Report and confusion matrix included

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/7c760bed-5617-48ce-bc45-b0076f092066" />


---

