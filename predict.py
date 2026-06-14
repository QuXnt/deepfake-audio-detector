import os
import argparse
import numpy as np
import tensorflow as tf

SAMPLING_RATE = 16000
MAX_DURATION = 3
MAX_SAMPLES = SAMPLING_RATE * MAX_DURATION

def load_and_preprocess_single_audio(file_path):
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
    
    return tf.expand_dims(spectrogram, 0) # Add batch dimension

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help='Path to the audio file to test.')
    parser.add_argument('--model_path', type=str, default=r"C:\mars ai\ps2\model\best_model.keras")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: File {args.audio_path} does not exist.")
        return

    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    print(f"Processing audio {args.audio_path}...")
    input_data = load_and_preprocess_single_audio(args.audio_path)
    
    print("Predicting...")
    OPTIMAL_THRESHOLD = 0.9999
    prediction_prob = model.predict(input_data)[0][0]
    
    is_real = prediction_prob >= OPTIMAL_THRESHOLD
    confidence = prediction_prob if is_real else 1 - prediction_prob
    
    print("\n" + "="*40)
    print("RESULT:")
    if is_real:
        print(f"Genuine (Human) - Confidence: {confidence*100:.2f}%")
    else:
        print(f"Deepfake (AI-Generated) - Confidence: {confidence*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
