# Efficient Gesture Recognition in Python

Real-time hand gesture recognition prototype designed for Edge AI devices using a lightweight **MobileNetV2 backbone** combined with the **Temporal Shift Module (TSM)** for efficient temporal modeling.

This project demonstrates how video-based gesture recognition can be deployed with low latency and low memory usage compared with heavier 3D CNN approaches.

---

## Project Overview

Hand gesture recognition is useful for:

- Smart home control
- Touchless interfaces
- Accessibility tools
- Robotics interaction
- AR / VR navigation

Traditional video recognition models can be computationally expensive. This project uses a lightweight architecture to improve deployment feasibility on constrained devices.

---

## Model Architecture

### Backbone
- **MobileNetV2**
- Lightweight CNN optimized for mobile / embedded hardware

### Temporal Module
- **Temporal Shift Module (TSM)**
- Enables temporal reasoning across multiple frames
- Much cheaper than 3D convolutions

### Pipeline

Webcam Input  
→ Frame Buffer (8 Frames)  
→ MobileNetV2 Backbone  
→ TSM Temporal Modeling  
→ Gesture Prediction

---

## Features

- Real-time webcam gesture recognition
- Temporal frame buffering
- Lightweight Edge AI architecture
- Python + PyTorch implementation
- OpenCV live video interface
- Expandable for dataset training / quantization

---
