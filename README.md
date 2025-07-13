# 🧠 AI-Driven Sign Language Recognition 🖐️

This project uses **MediaPipe** and a **CTR-GCN (Channel-wise Topology Refinement Graph Convolutional Network)** to recognize American Sign Language (ASL) gestures from a live webcam feed.

It detects hand keypoints using MediaPipe, processes them into graph data, and classifies gestures using a deep learning model trained on ASL.

---

## 📷 Demo Preview

> _(Add a screenshot or video link here later)_

---

## 🚀 Features

- ✋ Real-time hand detection using **MediaPipe**
- 🤖 Gesture recognition using a trained **CTR-GCN model**
- 🔡 Supports 26 ASL alphabets + `SPACE` + `DEL`
- 📦 Uses **PyTorch** for inference
- 💻 Works directly with a webcam (no browser needed)

---

## 🧠 How It Works

1. Captures frames from your webcam
2. Detects 21 hand landmarks using MediaPipe
3. Flattens the 3D coordinates to feed into the CTR-GCN model
4. Model predicts the gesture (A–Z, SPACE, DEL)
5. Displays predicted letter and confidence on screen

---

## 📦 Requirements

To run this project, you'll need the following Python libraries:

| Library       | Purpose                                 |
|---------------|-----------------------------------------|
| `torch`       | Deep learning framework (for CTR-GCN)   |
| `torchvision` | Helpful utilities (optional but useful) |
| `opencv-python` | Webcam and video stream handling     |
| `mediapipe`   | Hand tracking and keypoint detection    |
| `numpy`       | Numeric array manipulation              |
| `matplotlib`  | Visualization (loss curves, confusion matrix) |
| `seaborn`     | Enhanced plotting (for confusion matrix) |
| `scikit-learn`| Accuracy, precision, F1 score, train-test split |

---

### ✅ Install with pip:

```bash
pip install torch torchvision opencv-python mediapipe numpy matplotlib seaborn scikit-learn

## 📸 Screenshots

Here are some real-time frames captured during live ASL sign recognition using the CTR-GCN model:

<p float="left">
  <img src="screenshots/Screenshot_2025-04-06_214344.png" width="260"/>
  <img src="screenshots/Screenshot_2025-04-06_214408.png" width="260"/>
  <img src="screenshots/Screenshot_2025-04-06_214439.png" width="260"/>
  <img src="screenshots/Screenshot_2025-04-06_214455.png" width="260"/>
  <img src="screenshots/Screenshot_2025-04-06_214522.png" width="260"/>
  <img src="screenshots/Screenshot_2025-04-06_214544.png" width="260"/>
  <img src="screenshots/Screenshot_2025-04-06_214751.png" width="260"/>
  <img src="screenshots/Screenshot_2025-04-06_214842.png" width="260"/>
</p>

---

### 📊 Model Evaluation

#### Confusion Matrix

<img src="screenshots/Confusion_matrix.png" width="600"/>

---

### 🧪 Training Output Logs

<img src="screenshots/Training_Results.png" width="600"/>


