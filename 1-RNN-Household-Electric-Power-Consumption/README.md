
# ⚡ Household Electric Power Consumption
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

The goal of this project is to analyze and forecast household energy consumption using a **Recurrent Neural Network (RNN)** model.

## 📚 Table of Contents

- [Dependencies & Environment](#dependencies--environment)
- [Dataset Overview](#dataset-overview)
- [🧹 Data Preparation](#-data-preparation)
  - [1- Date-Time Parsing & Indexing](#1--date-time-parsing--indexing)
  - [2-⏱️ Resampling to Hourly Frequency](#2--resampling-to-hourly-frequency)
  - [3-🚧 Missing Value Imputation](#3--missing-value-imputation)
- [Feature Engineering](#-feature-engineering)
  - [Time-based Features](#-time-based-features)
  - [Feature Selection](#feature-selection)
- [📊 Visualizing Time-Based Patterns](#-visualizing-time-based-patterns)
  - [Daily Cycle of Energy Consumption](#daily-cycle-of-energy-consumption)
  - [Lag Correlation Analysis](#lag-correlation-analysis)
- [⚙️ Data Preparation for Modeling](#data-preparation-for-modeling)
  - [Train-Test Split](#-train-test-split)
  - [📏 Data Normalization / Scaling](#-data-normalization--scaling)
  - [Lookback Feature Creation](#lookback-feature-creation)
- [🏗️ RNN Model Architecture & Training](#rnn-model-architecture--training)
  - [📉 Training Loss](#-training-loss)
- [📈 Model Evaluation](#model-evaluation)
- [Predictions & Metrics](#-predictions--metrics)
  - [Predictions Metrics](#-predictions-metrics)
  - [Prediction Plot](#-prediction-plot)
  - [⚡ Global Active Power Prediction](#-global-active-power-prediction)


----
## Dependencies & Environment

### 📦 Main Dependencies

| Library | Purpose |
|----------|----------|
| **numpy**, **pandas** | Data manipulation and analysis |
| **matplotlib**, **seaborn** | Data visualization |
| **scikit-learn** | Data scaling and evaluation metrics |
| **tensorflow / keras** | Building and training the RNN model |
| **warnings** | Suppressing unnecessary warnings during training |

🧠 **Environment:** Google Colab (Python 3.x, TensorFlow 2.x)

🔧 Installation

If running locally, install dependencies with:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

```

----

## Dataset Overview

The **Household Electric Power Consumption** dataset contains minute-level measurements of electricity usage collected over almost four years (December 2006 → November 2010). It includes multiple electrical quantities and sub-metering values representing different areas of a household.

📘 Source: [Kaggle – Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

📅 Duration: Dec 2006 – Nov 2010

⏱️ Frequency: 1-minute intervals

📊 Total Records: 2,075,259 observations

----

## 🧹 Data Preparation

### 1- Date-Time Parsing & Indexing

- Combined the date and time columns into a single **datetime** column.
- Set the datetime column as the **index** of the dataset for easier time-series handling and resampling.

### 2- Resampling to Hourly Frequency
⏱️
- The original dataset contained **1-minute interval measurements** (~2 million rows).

- To simplify time-series analysis, data was **resampled to hourly frequency**, taking the **mean value** for each hour.

- This reduced the dataset to about **34,000 rows**, making it easier to analyze.

- Missing values (NaN) became more apparent and easier to handle after resampling.

### 3-🚧 Missing Value Imputation

Some hours had missing values after resampling.

- **Short gaps (≤ 3 hours)** were filled using **time-based interpolation.**

- **Medium gaps (>3h – 24h<)** and **large gaps (> 24 hours)** were removed.

- **Results:**
    - Detected gaps: 8 
    - Removed large gaps (>24h) : 406 hourly records
    - Interpolated gaps (≤ 3h): 49 cells filled
    - Rows removed (>3h – 24h<): 8
    - Total number of deleted rows : 414
    - Final dataset shape: (34,175 , 7)
    - Imputed cells: 49

---- 

## 🧠 Feature Engineering

### ⏰ Time-based Features

Added new columns to help the model learn daily and seasonal patterns:

hour, dayofweek, is_weekend, month

hour_sin, hour_cos → cyclical encoding of time (to represent periodic behavior)

These features help the model understand when energy consumption increases or decreases (e.g., by time of day or weekend).

**Source =>  [Cyclic Encoding: Sine and Cosine Transformations for Periodic Features](https://www.kaggle.com/discussions/general/491296)**

### Feature Selection

- Calculated correlation of all features with the target (`Global_active_power`).

- **Dropped low-correlation columns**: `dayofweek` and `month`  

----


## 📊 Visualizing Time-Based Patterns

Plotted average Global_active_power by hour to observe daily energy usage trends. 

#### Daily Cycle of Energy Consumption:

<img width="75%" height="300" alt="Image" src="https://github.com/user-attachments/assets/8b3a92a4-83a0-4804-821e-0ca7781659df" />

#### Lag Correlation Analysis

<img width="75%" height="310" alt="Image" src="https://github.com/user-attachments/assets/99236db7-f3ae-425b-8060-a57ab680d198" />

- Shows the correlation of past Global_active_power values (1–24 hours) with the target.

- **Helps identify which previous hours are most important for predicting current power usage.**

- Guides the choice of sequence length for the RNN model.

----

##  Data Preparation for Modeling

### 🧩 Train-Test Split

**Chronological split (no shuffling)** to preserve time order.
  - **Training set**: 70% 
  - **Validation set**: 15% 
  - **Test set**: 15%

### 📏 Data Normalization / Scaling

- **StandardScaler** was used to normalize features (mean = 0, std = 1).
- **The scaler was fitted only on the training set to avoid data leakage.**

### Lookback Feature Creation
- Created **sequences (time windows)** from the scaled data for RNN input.
- Input shape transformed from **2D** `(samples, features)`  → **3D** `(samples, timesteps, features)`
- Used a **24-hour lookback window** `(timesteps = 24)` to predict **1 hour ahead** `(horizon = 1)`. A new sample is created by sliding 1 hour each time.

----


## RNN Model Architecture & Training
🏗️
- **Model** : Sequential RNN with two layers: Two-layer **SimpleRNN** with **ReLU** activations and **Dropout** for regularization, followed by a **Dense** output layer for regression.
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Optimizer**: Adam (`learning_rate=0.001`) with **gradient clipping** (`clipnorm=2.0`) to prevent **exploding gradients** and ensure stable training.

**Notes**: `tanh and sigmoid` were tested, but `ReLU` gave better performance.

**Training Callbacks**:

- **EarlyStopping** : (patience=8)
- **ModelCheckpoint**: **Save best model** based on validation loss
- **ReduceLROnPlateau**: **Reduce LR** if val_loss plateaus

### 📈 Training Loss

- **Training and validation loss (MSE)** over epochs.
- **Early stopping** triggered when validation loss stopped improving, indicating stable training.

<img width="75%" height="393" alt="Image" src="https://github.com/user-attachments/assets/7e24c147-5f19-498f-b2f9-8034dbf8c824" />

----

## Model Evaluation

- Loaded the **best saved model** and evaluated on validation and test sets.

**📊 Model Results**

| Dataset     | MSE   | MAE   |
|--------------|-------|-------|
| Validation   | 0.326 | 0.396 |
| Test         | 0.257 | 0.355 |


**Shows that the model performs well and generalizes to unseen data.**

----
## 🔹 Predictions & Metrics

- Used the **best saved model** to make **predictions** on **train, validation, and test sets**.

- Converted predicted and actual values back to **original scale** using **inverse transform**.

- Calculated performance metrics for each set:
       - **MSE, RMSE, and MAE**

### 📊 Predictions Metrics

| Dataset     | MSE   | RMSE  | MAE   |
|------------|-------|-------|-------|
| Train      | 0.264 | 0.514 | 0.351 |
| Validation | 0.281 | 0.530 | 0.368 |
| Test       | 0.222 | 0.471 | 0.330 |

- The model performs well on both **seen (train)** and **unseen (validation/test)** data, showing good generalization.

### 📉 Prediction Plot

Shows true vs. predicted values on the test set :

<img width="75%" height="374" alt="Image" src="https://github.com/user-attachments/assets/2d33b423-0c86-44b2-9356-2d936c9639b7" />


### 📊 Global Active Power Prediction

**Shows true vs. predicted values for train, validation, and test sets.**

<img width="75%" height="420" alt="Image" src="https://github.com/user-attachments/assets/86045e5a-5f32-432c-992e-eecb8e6c6416" />


 

  






