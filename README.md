# Real-Time Object Detection and Live Stream Analysis Using Jetson Nano

## MINI PROJECT

### TEAM MEMBERS
- Ganesh Patidar (20214061)
- Hardik Kumar Singh (20214249)
- Divyanshu (20214317)
- Harsh Dave (20214534)

---

## CONTENTS
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

## Problem Statement
- LIVESTREAM CAMERA THROUGH JETSON NANO HARDWARE
- OBJECT DETECTION ON IMAGES, VIDEOS, AND LIVESTREAM CAMERA

## Introduction
This project presents a cutting-edge object detection system leveraging Jetson Nano's live stream capabilities for real-time surveillance and monitoring, facilitating prompt responses to detected objects or events.

The system stands out in accurately detecting and classifying objects in live streams through advanced deep learning algorithms, bolstering security and operational efficiency across diverse settings.

## Motivation
The motivation behind our mini project, "Real-Time Object Detection and Live Stream Analysis," stemmed from the pressing need to enhance security and safety measures in public transport systems. Recognizing the significance of real-time monitoring and crime detection, particularly for ensuring passenger safety, especially for vulnerable groups like females, we embarked on this project. Our aim was to leverage CCTV feeds to implement robust object detection algorithms and live stream analysis, empowering authorities with timely insights to prevent and address potential security threats effectively.

## Applications
- Surveillance and Security System
- Traffic Management
- Retail Analytics
- Industrial Automation
- Smart Cities
- Environmental Monitoring

## Proposed Work
- Setting Up Jetson Nano
- Live Streaming Implementation
- Data Collection and Model Training
- Exploration of Object Detection Models
- Analysis of Different Models

## Experimental Setup
1. **Setting Up the Jetson Nano:**
   - Flashed the Jetson Nano with the NVIDIA OS using Balena Etcher.
   - Utilized the JetPack SDK 4.4.0 for development environment setup.
   - Booted up the Jetson Nano for further configuration.

2. **Live Streaming Implementation:**
   - Employed OpenCV with CUDA environment for real-time video processing.
   - Enabled efficient video capture and frame processing for object detection.

3. **Data Collection and Model Training:**
   - Utilized `simple_image_download` to gather relevant training data.
   - Employed `labelImg` for image labeling, creating ground truth data.
   - Trained a custom YOLOv7 model using Google Colab for enhanced processing power.
   - Leveraged Google Colab's computational resources to train the model efficiently.

4. **Exploration of Object Detection Models:**
   - Investigated pre-trained models from TensorFlow Zoo.
   - Analyzed SSD ResNet50 640x640 and CenterNet ResNet101 FPNv1 512x512 models.
   - Evaluated their performance based on mean Average Precision (mAP) and inference time.

5. **Evaluation Metrics:**
   - Precision, Recall, mAP (mean Average Precision).
   - Precision = TP / (TP + FP), Recall = TP / (TP + FN).
   - mAP = Average of AP across all classes.

6. **Results:**
   - Working of Centernet Resnet101_FPNv1.
   - Working of Centernet Resnet101_FPNv1.

## Result Analysis
### Accuracy Comparison
- CenterNet ResNet-101 exhibits the lowest mAP among the three models, indicating lower overall accuracy in object detection.
- YOLOv7 (Custom) achieves the highest mAP, demonstrating superior performance in accurately detecting and classifying objects.
- SSD ResNet-50 640x640 has a similar mAP to CenterNet ResNet-101, suggesting comparable accuracy but with a faster inference time.

### Inference Time Trade-off
- CenterNet ResNet-101 boasts the fastest inference time, making it suitable for real-time applications where speed is critical.
- YOLOv7 (Custom) has a significantly higher inference time, potentially limiting its use in scenarios demanding low latency.
- SSD ResNet-50 640x640 offers a balance between accuracy and speed with a moderate inference time.

### Example Results
![Comparison Graph](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/801819ba-2871-4f6b-8425-f33bab728d98)
![Result Image_1](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/97cc35d7-9780-44a0-89d6-f23a6198ab97)
![Result Image_2](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/b056f45f-4228-430f-8a54-67c2858e3232)
![Result Image_3](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/e6c20bff-9184-4b23-92a8-c9f266b7d69c)
![Result Image_4](https://github.com/Hardik-111/livestream_object_detection/assets/89783619/b43ee2dd-9ef5-426a-a1b1-efff22b16faa)


## Challenges
- Proxy Configuration Challenges
- Package Installation Issues

- SSL Wrong Version Number
- Python Version Compatibility
- Extended Training Epochs
- Anaconda Incompatibility on Jetson Nano
- Jetson Shutdown Instances

## Future Work
- Performance Optimization
- Integration with Cloud Services
- Real-time Alerts and Notifications
- User Interface Enhancement
- Integration with IoT Devices

## References
1. Abadi, M., et al. TensorFlow Model Zoo. Available online: [TensorFlow Model Zoo]([https://github.com/tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)) (Accessed: May 2, 2024).
2. Liu, W., Anguelov, D., et al. SSD: Single Shot Multibox Detector. European Conference on Computer Vision (ECCV) 9905 (2016), 21–37.
3. Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. You Only Look Once: Unified, Real-Time Object Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence 38, 1 (2016), 78–87.
4. Wang, J., Chen, K., Wang, S., and Hoi, S. YOLOv7: A Trainable Bag of Freebies for Real-Time Object Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 12 (2021), 4239–4249.
5. [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

---

Thank you!
