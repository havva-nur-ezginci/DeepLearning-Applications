
# ⚡ Household Electric Power Consumption
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

The goal of this project is to analyze and forecast household energy consumption using a **Recurrent Neural Network (RNN)** model.

# Table of Content

----

## Dataset Overview

The **Household Electric Power Consumption** dataset contains minute-level measurements of electricity usage collected over almost four years (December 2006 → November 2010). It includes multiple electrical quantities and sub-metering values representing different areas of a household.

📘 Source: [Kaggle – Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

📅 Duration: Dec 2006 – Nov 2010

⏱️ Frequency: 1-minute intervals

📊 Total Records: 2,075,259 observations

----

## Data Preparation

### 1- Date-Time Parsing & Indexing

- Combined the date and time columns into a single **datetime** column.
- Set the datetime column as the **index** of the dataset for easier time-series handling and resampling.

### 2- ⏱️ Resampling to Hourly Frequency

- The original dataset contained **1-minute interval measurements** (~2 million rows).

- To simplify time-series analysis, data was **resampled to hourly frequency**, taking the **mean value** for each hour.

- This reduced the dataset to about **34,000 rows**, making it easier to analyze.

- Missing values (NaN) became more apparent and easier to handle after resampling.

### 3- Missing Value Imputation

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

## Feature Engineering

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

#### **Daily Cycle of Energy Consumption:** 

<img width="989" height="490" alt="Image" src="https://github.com/user-attachments/assets/8b3a92a4-83a0-4804-821e-0ca7781659df" />

#### 📊 Lag Correlation Analysis

<img width="826" height="300" alt="Image" src="https://github.com/user-attachments/assets/99236db7-f3ae-425b-8060-a57ab680d198" />

- Shows the correlation of past Global_active_power values (1–24 hours) with the target.

- **Helps identify which previous hours are most important for predicting current power usage.**

- Guides the choice of sequence length for the RNN model.

----

## Data Preparation

### 🧩 Train-Test Split

**Chronological split (no shuffling)** to preserve time order.
  - **Training set**: 70% 
  - **Validation set**: 15% 
  - **Test set**: 15%

### Data Normalization / Scaling

- **StandardScaler** was used to normalize features (mean = 0, std = 1).
- **The scaler was fitted only on the training set to avoid data leakage.**

### Lookback Feature Creation
- Created **sequences (time windows)** from the scaled data for RNN input.
- Input shape transformed from **2D** `(samples, features)`  → **3D** `(samples, timesteps, features)`
- Used a **24-hour lookback window** `(timesteps = 24)` to predict **1 hour ahead** `(horizon = 1)`







 

  
