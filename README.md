# 🪧🚗🤖 Traffic Sign Classification Model Using LE-NET Architecture for Self-Driving Cars

This project implements a **Convolutional Neural Network (CNN)** based on the classic **LeNet-5 architecture** to classify traffic sign images into one of 43 classes.  
It is designed as a proof-of-concept for traffic sign recognition — a key component of autonomous driving systems — using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

---

## 📑 Table of Contents

* [🚀 Project Overview](#-project-overview)
* [🗃️ Dataset Overview](#-dataset-overview)
* [⚙️ Setup](#-setup)
* [📚🗂️ Import Libraries and Dataset](#-import-libraries-and-dataset)
* [🔍🖼️ Image Exploration (EDA)](#-image-exploration-eda)
* [🧹 Data Preparation](#-data-preparation)
* [🧠🚀 Model Training](#-model-training)
* [📈🧪 Model Evaluation and Inference](#-model-evaluation-and-inference)
* [✅ Results](#-conclusions)
* [🤝 Contributing](#-contributing)


---

## 🚀 Project Overview

Traffic sign recognition is essential for the safety and effectiveness of self-driving vehicles.  
This project:

* Trains a CNN (LeNet) to classify traffic signs into 43 categories.
* Uses the German Traffic Sign dataset.
* Demonstrates an end-to-end pipeline: data exploration, preparation, training, evaluation, and saving the trained model.

---

## 🗃️ Dataset Overview

* 📍 **Source:** [Kaggle — Traffic Sign Classification Model Using LE-NET](https://www.kaggle.com/datasets/mehardeepsandhu/traffic-sign-classification-model-using-le-net)  
* 🖼️ **Classes:** 43 different traffic signs (e.g., speed limits, stop, yield, no entry, caution signs).  
* 📊 **Datasets:** Training, Validation, Test  
* 🗂️ Each image is labeled with its correct traffic sign class.  
* Images vary in lighting, size, and orientation, requiring preprocessing and robust modeling.

---

## ⚙️ Setup

### 🔧 Requirements

* Python 3.7+
* Libraries:
  * tensorflow / keras
  * numpy
  * matplotlib
  * seaborn
  * sklearn

### 🧰 Install dependencies:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### 📂 Clone this repo:

```bash
https://github.com/Mehardeep79/Traffic-Signals-LENET-Classification-Model.git
cd Traffic-Signals-LENET-Classification-Model
```

## 📚🗂️ Import Libraries and Dataset

We import all required Python libraries, load the GTSRB dataset, and split it into training, validation, and test sets.

---

## 🔍🖼️ Image Exploration (EDA)

We perform exploratory data analysis to:
- Inspect a few example images from each dataset.
- Confirm that labels match the images.
- Understand the distribution of traffic sign classes.

---

## 🧹 Data Preparation

Before feeding data into the CNN:
- Images are resized to a uniform size compatible with LeNet.
- Pixel values are normalized to improve convergence.
- Images are reshaped to include the channel dimension.

---

## 🧠🚀 Model Training

We implement and train a **LeNet-5 inspired CNN**, which consists of:
- Two convolution + pooling layers
- Flattening and fully connected dense layers
- Softmax output for 43 classes
- **Training Parameters:**
    - Learning Rate -> 0.001
    - Epochs -> 500
    - Batch Size -> 500


The model is trained on the training set and validated on the validation set.

---

## 📈🧪 Model Evaluation and Inference

After training:
- The model is evaluated on the test set.
- Accuracy and confusion matrix are computed.
- Sample predictions are visualized to assess which signs were classified correctly or misclassified.

---

## ✅ Conclusions

- The LeNet-based model effectively learns to classify traffic signs.
- Achieved 91% accuracy on the test set, showing feasibility of this approach.
- Improvements could include:
  - Data augmentation
  - Using deeper architectures (e.g., ResNet, MobileNet)
  - Hyperparameter tuning

---

## 🤝 Contributing

Contributions are welcome!  
If you have suggestions for improvements, feel free to fork the repo, enhance it, and submit a pull request.



