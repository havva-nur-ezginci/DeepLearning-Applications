"# DeepLearning-Applications" 
---
# DL Resources

https://poloclub.github.io/cnn-explainer/ 

https://netron.app/

---

# Table of Contents
- [1-CNN-Dogs-and-Cats-Classification](#dog---cat-classification-with-cnn)


---

# Dog🐶 - Cat🐱 Classification with CNN
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
