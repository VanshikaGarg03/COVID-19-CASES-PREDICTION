# 🧠 COVID-19 Chest X-Ray Classification using CNN

This project is a deep learning model that detects **COVID-19** from chest X-ray images using **Convolutional Neural Networks (CNN)**. It is built using **TensorFlow**, **Keras**, and trained on a real-world dataset of labeled chest scans.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/135VosS6CCCpLqmWqc_09IOamQ60psM9S#scrollTo=FYLqW-q-pGjg&printMode=true)

---

## 📌 Project Objective
The goal is to classify X-ray images into two categories:
- **COVID-19 Positive**
- **Normal (Healthy)**

This kind of model can assist in rapid screening where RT-PCR facilities are limited.

---

## 📁 Dataset
- Used from: [RishitToteja/Chext-X-ray-Images-Data-Set](https://github.com/RishitToteja/Chext-X-ray-Images-Data-Set)
- Directory Structure:
├── train
│ ├── COVID19
│ └── NORMAL
└── test
├── COVID19
└── NORMAL
- Preprocessed using `ImageDataGenerator` with data augmentation

---

## 🧠 Model Architecture

- **2 Convolutional layers** (32 & 64 filters)
- **MaxPooling** after each Conv layer
- **Dropout layers** to prevent overfitting
- **Flatten + Dense (256)** fully connected layer
- **Output layer** with sigmoid activation for binary classification

Compiled with:
```python
optimizer = Adam(learning_rate=0.001)
loss = 'binary_crossentropy'
metrics = ['accuracy']
