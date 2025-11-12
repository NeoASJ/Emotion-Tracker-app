import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# ------------------------------ #
# 🎧 Streamlit App Title
# ------------------------------ #
st.set_page_config(page_title="Emotion Recognition", page_icon="🎤")
st.title("🎧 Speech Emotion Recognition")
st.write("Upload an audio file and select which model (dataset) to use for emotion detection.")

# ------------------------------ #
# 🔧 Model Selection
# ------------------------------ #
model_choice = st.selectbox(
    "Select Model:",
    ["RAVDESS", "CREMA-D", "EMODB"]
)

# ------------------------------ #
# 🧠 Emotion Labels (custom for each dataset)
# ------------------------------ #
emotion_labels = {
    "RAVDESS": ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
    "CREMA-D": ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust'],
    "EMODB":   ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'sadness', 'neutral']
}

# ------------------------------ #
# 📦 Load selected model
# ------------------------------ #
model_path = f"emotion_model_sep_{model_choice.lower()}.h5"
model = load_model(model_path)
st.success(f"✅ Loaded {model_choice} model")

# ------------------------------ #
# 🎵 Audio Upload Section
# ------------------------------ #
audio_file = st.file_uploader("Upload your .wav file", type=["wav"])

def extract_features(file, n_mfcc=40):
    y, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfcc

# ------------------------------ #
# 🎯 Prediction
# ------------------------------ #
if audio_file is not None:
    st.audio(audio_file)
    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Get emotion name
    emotion_name = emotion_labels[model_choice][emotion_index] if emotion_index < len(emotion_labels[model_choice]) else "Unknown"

    st.subheader("🎤 Predicted Emotion:")
    st.write(f"**Emotion:** {emotion_name} ({confidence*100:.2f}% confidence)")
    st.progress(float(confidence))

    st.write("🔢 Prediction Probabilities:")
    st.json({emotion_labels[model_choice][i]: float(prediction[0][i]) for i in range(len(prediction[0]))})
