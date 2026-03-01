# 🎵 Music Genre Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/Library-TensorFlow-orange?logo=tensorflow)
![Librosa](https://img.shields.io/badge/Audio-Librosa-green?logo=python)

## 📖 Overview
**Task 11** explores Audio Signal Processing. We built a Deep Learning model (CNN) to classify music tracks into 10 genres (Rock, Jazz, Pop, etc.) using the **GTZAN Dataset**.

## ⚙️ Workflow
1.  **Feature Extraction:**
    * Loaded 30-second audio clips using `librosa`.
    * Sliced each track into **10 segments** (3 seconds each) to increase dataset size.
    * Converted audio waves into **MFCCs** (Mel-Frequency Cepstral Coefficients) — essentially "images" of sound.
2.  **Model Architecture:**
    * Built a **Convolutional Neural Network (CNN)** using Keras.
    * Input: MFCC Heatmaps (130 time steps x 13 coefficients).
    * Layers: 3 Conv2D layers with MaxPooling and BatchNormalization.
3.  **Inference:**
    * Implemented a **Majority Voting System**: The model listens to 10 different parts of a new song and votes on the genre to improve accuracy.

## 📊 Results
* **Training Accuracy:** ~75%
* **Test Accuracy:** ~60-70% (Typical for this dataset without advanced Transfer Learning).
* **Key Insight:** Genres like *Rock* and *Country* often overlap in instrumentation, causing misclassifications, while distinct genres like *Classical* are easier to detect.
