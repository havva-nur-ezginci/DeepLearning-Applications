# 📰RNN-News Topic Classification 
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data)

The goal of this project is to build an **RNN-based model** that automatically classifies **news articles** from the AG News dataset into four **predefined categories.**


----
## Dependencies & Environment

### 📦 Main Dependencies

| Library                 | Purpose                                                           |
| ----------------------- | ----------------------------------------------------------------- |
| **numpy, pandas**       | Data manipulation and preprocessing                               |
| **matplotlib, seaborn** | Visualizations (plots, ROC curves, confusion matrix)              |
| **tensorflow / keras**  | Building, training, and tuning the RNN model                      |
| **keras-tuner**         | Hyperparameter optimization (Random Search)                       |
| **scikit-learn**        | Evaluation metrics (classification report, ROC, confusion matrix) |


🧠 **Environment:** Kaggle Notebooks using the GPU T4 (Python 3.x, TensorFlow 2.x)

🔧 Installation

If running locally, install dependencies with:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras-tuner
```

---

# Dataset Overview

The AG News dataset contains news articles labeled into four main categories: World, Sports, Business, and Science/Technology. Each sample consists of a title and a short description, providing concise textual information for classification tasks.

**📘 Source**: [AG News – Academic news search engine ComeToMyHead](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data)

**📚 Categories**: 4 predefined classes

**📁 Files**: train.csv and test.csv

**📝 Content**: Class index, title, and description

**🔢 Total Samples**: 127,600 (120,000 training + 7,600 test)

---

# Text Preprocessing

The Title and Description fields were merged into a single text input and prepared for the RNN model using tokenization.

**Process:**

1. Merged title + description.
2. Applied a tokenizer with a 10,000-word vocabulary and an <OOV> token for unseen words.
3. Converted text into integer sequences.
4. Applied padding and truncation (padding='post', truncating='post') to ensure uniform sequence length (maxlen = 100).

---

# RNN Model Architecture 🏗️

The model uses a **SimpleRNN** structure to learn patterns from the text.

- Embedding Layer: Turns each word into a vector.
- SimpleRNN Layer: Processes the sequence step-by-step and learns the meaning of the text.
- Dropout: Reduces overfitting.
- Dense Output Layer: Softmax layer that predicts the correct news category.

**The model was trained with the Adam optimizer, gradient clipping for stability, and categorical crossentropy for multi-class classification.**

---

# Hyperparameter Search ⚙️🔎

To improve model performance, a** Random Search (via Keras Tuner)** was used to **find the best hyperparameters.**

The search optimized:

1. Embedding dimension
2. Number of RNN units
3. Dropout rate
4. Gradient clipping value

- A total of **30 different model** configurations were tested.
- Each configuration was trained once, and **early stopping** was applied to avoid unnecessary training.
- This process helped identify the most effective architecture before training the final model.

## Best Hyperparameters 🎯

After the search, the tuner found the best configuration with a validation accuracy of 0.888.

**Selected Hyperparameters:**

- Embedding Dimension: 128
- RNN Units: 160
- Dropout Rate: 0.3
- Clipnorm: 1.0

---

# Model Evaluation 📊

After selecting the **best hyperparameters**, the final model was**evaluated on the test set.**

**Test Performance**:

- **Accuracy:** 0.8915

- **Loss:** 0.3542

- **AUC:** 0.9756

These results show that the model generalizes well across the four news categories.

## ROC Curves

ROC curves were plotted for each class using the predicted probabilities.
They show how well the model distinguishes each category across different thresholds.

<img width="%80" height="%80" alt="image" src="https://github.com/user-attachments/assets/7c760bed-5617-48ce-bc45-b0076f092066" />

## Classification Report

| Class                | Precision | Recall | F1-Score | Support |
| -------------------- | --------- | ------ | -------- | ------- |
| World                | 0.91      | 0.89   | 0.90     | 1900    |
| Sports               | 0.95      | 0.95   | 0.95     | 1900    |
| Business             | 0.84      | 0.89   | 0.87     | 1900    |
| Science              | 0.89      | 0.86   | 0.87     | 1900    |
| **Overall Accuracy** | —         | —      | **0.90** | 7600    |


The model performs strongest on Sports and World, with slightly lower performance on Business and Science, which typically contain more overlapping terminology.

## Confusion Matrix

A confusion matrix was generated to visualize how well the model distinguishes between the four categories.

<img width="%80" height="%80" alt="image" src="https://github.com/user-attachments/assets/b9154848-44b9-416a-9b86-f881864d6df4" />

