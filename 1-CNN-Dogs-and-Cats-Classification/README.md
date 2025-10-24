# Dog🐶 and Cat🐱 Classification
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data)

🧠 Deep Learning Model

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs.
The model was trained on the Kaggle Cats vs Dogs dataset using Google Colab.

---

## Step 1: Dataset Download

The dataset for cats and dogs was obtained from Kaggle. First, the Kaggle API key was uploaded to Colab (kaggle.json) to authenticate the account. Then, the dataset was downloaded using the Kaggle CLI:
```sh 
# Upload kaggle.json to Colab first
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

After preprocessing, the data distribution was as follows:

* Training images: 6,404
* Validation images: 1,601
* Test images: 2,023

This setup ensured efficient memory usage on Colab while providing diverse and augmented data for model training.




---

🔹 Model Architecture

The CNN model consists of:


---
