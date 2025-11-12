<div align="center">

# 🎙️ <span style="font-size:42px; font-weight:800;">Emotion Recognition from Speech</span>

### <span style="font-size:20px;">A Deep Learning System for Multilingual Emotion Detection using Audio Signal Processing & Cloud Deployment</span>

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU-orange?logo=tensorflow)
![Librosa](https://img.shields.io/badge/Audio-Librosa-lightgrey?logo=soundcloud)
![Docker](https://img.shields.io/badge/Containerized-Docker-blue?logo=docker)
![Cloud Run](https://img.shields.io/badge/Deployed-Google%20Cloud%20Run-4285F4?logo=google-cloud)
</div>

---

## 🧭 <span style="font-size:26px;">Overview</span>

This project implements a **deep learning–based emotion recognition system** that classifies human emotions from voice recordings.  
It integrates **audio preprocessing**, **neural model training**, and **cloud deployment** into a **production-grade pipeline**.

Trained on **three benchmark emotion datasets — RAVDESS, CREMA-D, and EMO-DB**, the models leverage **GPU acceleration (CUDA)** for efficient feature extraction and optimization.

The final Streamlit app is **Dockerized and deployed on Google Cloud Run**, supporting real-time inference.

---

⚡ <span style="font-size:26px;">Key Highlights</span>

✅ **High-Fidelity Audio:** Trained exclusively on `.wav` files  
🔄 **FFmpeg Pipeline:** Converts `.mp3` → `.wav` seamlessly  
🎛 **Audio Features:** MFCCs, Chroma, Spectral Centroid, Zero Crossing Rate, RMS Energy  
🧠 **Three Deep Models:**
- CNN — *RAVDESS*  
- BiLSTM — *CREMA-D*  
- CNN + BiLSTM Hybrid — *EMO-DB*
🚀 **GPU-Accelerated** training (TensorFlow-GPU backend)  
🌐 **Cloud Deployment** via Docker + GCP Cloud Run  
💰 **Budget Control:** Configured cost limits with usage alerts  

---

🧩 <span style="font-size:26px;">Workflow Overview</span>

```
 ┌──────────────────────────────┐
 │      Audio Dataset           │
 │     (.wav files)      │
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   FFmpeg Conversion          │
 │   (.mp3 → .wav)              │
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   Feature Extraction          │
 │   (MFCC, Chroma, Spectral)    │
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   Model Training              │
 │   (CNN, BiLSTM, CNN+BiLSTM)  │
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   Model Evaluation &          │
 │   Confusion Matrix Generation │
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   Model Export                │
 │   (.h5 + .pkl files)          │
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   Docker Containerization     │
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   Google Cloud Run Deployment│
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │   Streamlit Web App           │
 │   Real-Time Emotion Detection │
 └──────────────────────────────┘


---

📚 Dataset Details

| **Dataset** | **Samples** | **Emotions** | **Language** | **Notes** |
|--------------|--------------|---------------|---------------|-----------|
| 🎭 RAVDESS | 2880 | 8 (calm, happy, sad, angry, fearful, disgust, surprised, neutral) | English | Acted emotional speech |
| 🎙 CREMA-D | 7442 | 6 (angry, disgust, fear, happy, sad, neutral) | English | Crowd-acted dataset |
| 🎤 EMO-DB | 583 | 7 (anger, boredom, disgust, fear, happy, neutral, sad) | German | High-quality emotional dataset |

Each dataset was processed and saved in `.pkl` format:

```python
ravdess_df.to_pickle("ravdess_features.pkl")
crema_df.to_pickle("crema_features.pkl")
emodb_df.to_pickle("emodb_features.pkl")
```

---

## 🎚 <span style="font-size:26px;">Audio Preprocessing</span>

All inputs are **.wav files**. For any `.mp3` data, automatic conversion is performed with **FFmpeg**:

```bash
for f in *.mp3; do ffmpeg -i "$f" "${f%.mp3}.wav"; done
```

### Extracted Features (Librosa):
- MFCCs  
- Chroma STFT  
- Spectral Centroid & Bandwidth  
- Zero-Crossing Rate  
- Root Mean Square Energy  

---

## 🧠 <span style="font-size:26px;">Model Training</span>

| **Model File** | **Dataset** | **Architecture** | **Epochs** | **Input Features** |
|----------------|--------------|------------------|-------------|--------------------|
| `emotion_model_ravdess.h5` | RAVDESS | CNN | 50 | MFCC + Chroma |
| `emotion_model_crema.h5` | CREMA-D | BiLSTM | 60 | MFCC |
| `emotion_model_emodb.h5` | EMO-DB | CNN + BiLSTM | 40 | MFCC + Spectral |

📊 Confusion matrices (`*_confusion_matrix.png`) visualize accuracy for each trained model.

---

## 🗂️ <span style="font-size:26px;">Project Structure</span>

```
emotion-tracker/
│
├── app.py                          # Streamlit inference app
├── main.py                         # Feature extraction & model training
├── data.ipynb                      # EDA & preprocessing notebook
├── Dockerfile                      # Docker build file
├── requirements.txt                # Dependencies list
│
├── ravdess_features.pkl
├── crema_features.pkl
├── emodb_features.pkl
│
├── emotion_model_ravdess.h5
├── emotion_model_crema.h5
├── emotion_model_emodb.h5
│
├── *_confusion_matrix.png          # Model evaluation plots
├── .dockerignore
├── .gitignore
└── zipfolder/                      # Archived artifacts
```

---

## 💻 <span style="font-size:26px;">Deployment Instructions</span>

### 🧩 Local Run
```bash
docker build -t emotion-recognition-app .
docker run -p 8501:8501 emotion-recognition-app
```
➡️ Access locally at: **[http://localhost:8501](http://localhost:8501)**

---

### ☁️ Deploy to Google Cloud Run
```bash
docker tag emotion-recognition-app gcr.io/global-tine-477418-b7/emotion-recognition-app:v1
gcloud auth configure-docker
docker push gcr.io/global-tine-477418-b7/emotion-recognition-app:v1

gcloud run deploy emotion-recognition-app   --image gcr.io/global-tine-477418-b7/emotion-recognition-app:v1   --platform managed   --region asia-south1   --allow-unauthenticated   --memory 2Gi   --port 8501
```

🌐  Cloud URL:  
`https://emotion-recognition-app-684612753531.asia-south1.run.app/`
## 🚀 Live Demo

My deployed app is available here:

🔗 [https://emotion-recognition-app-684612753531.asia-south1.run.app](https://emotion-recognition-app-684612753531.asia-south1.run.app)

---

## ⚙️ <span style="font-size:26px;">GPU Configuration</span>

- Framework: **TensorFlow-GPU (CUDA 11.x + cuDNN)**  
- GPU usage monitored via:
  ```bash
  nvidia-smi
  ```
- Docker runs on CPU in production; GPU used during training locally.

---



## 🔮 <span style="font-size:26px;">Future Enhancements</span>

- 🎤 Add microphone-based live emotion detection  
- 🌍 Expand multilingual support (Hindi)  
- ⚙️ Add FastAPI RESTful API endpoint  
- 🔁 Implement CI/CD with GitHub Actions  
- 🧠 Experiment with Transformer-based models (Wav2Vec2, HuBERT)

---


---
