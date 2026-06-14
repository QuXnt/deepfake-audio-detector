import streamlit as st
import tensorflow as tf
import os
import tempfile
import time

# UI Settings
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    .metric-container {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    p {
        text-align: center;
        font-size: 1.1rem;
        color: #a0aab2;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Deepfake Audio Detector")
st.markdown("<p>Upload an audio file to determine if it is Genuine (Human) or Deepfake (AI-Generated).</p>", unsafe_allow_html=True)

# Parameters
SAMPLING_RATE = 16000
MAX_DURATION = 3
MAX_SAMPLES = SAMPLING_RATE * MAX_DURATION
OPTIMAL_THRESHOLD = 0.9999 # Calibrated from EER

@st.cache_resource
def load_model():
    # PATCH: Kaggle saves Keras 3 models with 'quantization_config' which older Keras versions reject.
    # We intercept layer deserialization and pop this argument to prevent crashing.
    import keras
    
    class PatchedDense(keras.layers.Dense):
        def __init__(self, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(**kwargs)
            
    class PatchedConv2D(keras.layers.Conv2D):
        def __init__(self, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(**kwargs)

    keras.utils.get_custom_objects()['Dense'] = PatchedDense
    keras.utils.get_custom_objects()['Conv2D'] = PatchedConv2D

    model_path = r"model/best_model.keras"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

def preprocess_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    
    audio_len = tf.shape(audio)[0]
    if audio_len < MAX_SAMPLES:
        padding = [[0, MAX_SAMPLES - audio_len]]
        audio = tf.pad(audio, padding, "CONSTANT")
    else:
        audio = audio[:MAX_SAMPLES]
        
    stft = tf.signal.stft(audio, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    spectrogram = tf.expand_dims(spectrogram, -1)
    
    return tf.expand_dims(spectrogram, 0)

uploaded_file = st.file_uploader("Choose a .wav file", type=['wav'])

if model is None:
    st.error("Model not found. Please train the model first and save it to `model/best_model.keras`.")
elif uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Audio"):
        with st.spinner("Analyzing audio features..."):
            # Save uploaded file to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            try:
                # Preprocess and predict
                input_data = preprocess_audio(tmp_path)
                
                # Simulate loading for better UX
                time.sleep(1)
                
                prediction_prob = model.predict(input_data)[0][0]
                is_real = prediction_prob >= OPTIMAL_THRESHOLD
                confidence = prediction_prob if is_real else 1 - prediction_prob
                
                # Display Results
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.subheader("Classification")
                    if is_real:
                        st.markdown("<h2 style='color: #00fa9a;'>Genuine 👤</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h2 style='color: #ff4b4b;'>Deepfake 🤖</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col2:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.subheader("Confidence")
                    st.markdown(f"<h2>{confidence*100:.2f}%</h2>", unsafe_allow_html=True)
                    st.progress(float(confidence))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
