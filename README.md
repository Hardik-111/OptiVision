# Real-Time Object Detection and Live Stream Analysis Using Jetson Nano

## 🎯 MINI PROJECT

### 👥 TEAM MEMBERS
- **Ganesh Patidar** (20214061)
- **Hardik Kumar Singh** (20214249)
- **Divyanshu** (20214317)
- **Harsh Dave** (20214534)

---

## 📌 CONTENTS
- [Problem Statement](#problem-statement)
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Applications](#applications)
- [Proposed Work](#proposed-work)
- [Experimental Setup](#experimental-setup)
- [Result Analysis](#result-analysis)
- [Challenges](#challenges)
- [Future Work](#future-work)
- [References](#references)

---

## 📢 Problem Statement
- **Livestream Camera Integration with Jetson Nano Hardware**
- **Object Detection on Images, Videos, and Livestream Feeds**

## 📖 Introduction
This project implements a **real-time object detection system** using Jetson Nano, leveraging deep learning algorithms for accurate and efficient object classification. It enhances surveillance, security, and operational efficiency in various applications.

## 💡 Motivation
The inspiration for this project stems from the critical need to improve **security measures** in **public transport systems**. By leveraging **real-time CCTV feeds**, we aim to provide an **automated surveillance system** that ensures passenger safety, particularly for vulnerable groups. Our **goal** is to enable authorities to detect potential security threats **proactively**.

## 🚀 Applications
- **Surveillance and Security Systems**
- **Traffic Management**
- **Retail Analytics**
- **Industrial Automation**
- **Smart Cities**
- **Environmental Monitoring**

## 🔍 Proposed Work
- **Jetson Nano Setup**
- **Live Stream Implementation**
- **Data Collection & Model Training**
- **Evaluation of Object Detection Models**
- **Performance Analysis of Different Models**

## 🛠 Experimental Setup
### 1️⃣ **Setting Up Jetson Nano**
- Flashed the **NVIDIA OS** using **Balena Etcher**.
- Installed **JetPack SDK 4.4.0** for development.
- Booted Jetson Nano and configured the environment.

### 2️⃣ **Live Streaming Implementation**
- Utilized **OpenCV with CUDA** for optimized real-time video processing.
- Enabled efficient video capture and frame-by-frame object detection.

### 3️⃣ **Data Collection & Model Training**
- Collected data using `simple_image_download`.
- Labeled images using `labelImg`.
- Trained a **YOLOv7** model using **Google Colab** for improved computational performance.

### 4️⃣ **Evaluation of Object Detection Models**
- Compared **TensorFlow Model Zoo** models:
  - **SSD ResNet50 640x640**
  - **CenterNet ResNet101 FPNv1 512x512**
- Evaluated based on **mean Average Precision (mAP)** and **inference time**.

### 5️⃣ **Performance Metrics**
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **mAP** = Average of AP across all classes

## 📊 Result Analysis
### ✅ **Accuracy Comparison**
| Model | mAP (Accuracy) |
|--------|--------------|
| **CenterNet ResNet-101** | **Low** |
| **SSD ResNet-50** | **Moderate** |
| **YOLOv7 (Custom)** | **High** |

### ⚡ **Inference Time Trade-offs**
- **Fastest:** CenterNet ResNet-101 (Low accuracy, high speed)
- **Balanced:** SSD ResNet-50 (Moderate speed & accuracy)
- **Most Accurate:** YOLOv7 (High accuracy, slower inference)

### 📷 **Example Results**
![Comparison Graph](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/801819ba-2871-4f6b-8425-f33bab728d98)

![Result Image_1](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/97cc35d7-9780-44a0-89d6-f23a6198ab97)

![Result Image_2](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/b056f45f-4228-430f-8a54-67c2858e3232)

![Result Image_3](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/e6c20bff-9184-4b23-92a8-c9f266b7d69c)

![Result Image_4](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/b43ee2dd-9ef5-426a-a1b1-efff22b16faa)

## 🛑 Challenges
- **Proxy Configuration Issues**
- **Package Installation Errors**
- **SSL Wrong Version Number**
- **Python Version Conflicts**
- **Extended Training Time**
- **Jetson Nano Compatibility Issues**
- **Unexpected Shutdowns During Execution**

## 🔮 Future Work
- **Performance Optimization**
- **Cloud Integration**
- **Real-time Alerts & Notifications**
- **Enhanced User Interface**
- **IoT Device Integration**

## 📚 References
1. **Abadi, M. et al.** [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
2. **Liu, W., Anguelov, D., et al.** SSD: Single Shot Multibox Detector, ECCV (2016)
3. **Redmon, J., et al.** YOLO: Unified, Real-Time Object Detection, IEEE TPAMI (2016)
4. **Wang, J., et al.** YOLOv7: Trainable Bag of Freebies, IEEE TPAMI (2021)
5. **[PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)**

---

🚀 **Thank you!** We appreciate your time in reviewing our project! 🎯

