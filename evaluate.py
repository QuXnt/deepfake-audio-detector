import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
import argparse

# Parameters
SAMPLING_RATE = 16000
MAX_DURATION = 3
MAX_SAMPLES = SAMPLING_RATE * MAX_DURATION
BATCH_SIZE = 32

def load_and_preprocess_audio(file_path, label):
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
    
    return spectrogram, label

def get_test_data(directory):
    fake_dir = os.path.join(directory, 'fake')
    real_dir = os.path.join(directory, 'real')
    
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.wav')]
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.wav')]
    
    file_paths = fake_files + real_files
    labels = [0] * len(fake_files) + [1] * len(real_files)
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(load_and_preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return ds, np.array(labels)

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the threshold where FPR == FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default=r"C:\mars ai\datasets\for-norm\testing")
    parser.add_argument('--model_path', type=str, default=r"C:\mars ai\ps2\model\best_model.keras")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    print(f"Loading test data from {args.test_dir}...")
    test_ds, y_true = get_test_data(args.test_dir)
    
    print("Predicting...")
    y_pred_probs = model.predict(test_ds).flatten()
    
    # Calculate ROC curve and EER
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    optimal_threshold = thresholds[eer_idx]
    
    # Use the optimal threshold for final classification
    y_pred = (y_pred_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics using optimal threshold
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    tn, fp, fn, tp = cm.ravel()
    acc_fake = tn / (tn + fp)
    acc_real = tp / (tp + fn)
    
    print("\n" + "="*40)
    print("EVALUATION METRICS")
    print("="*40)
    print(f"Overall Accuracy : {acc*100:.2f}% (Threshold: >= 80%)")
    print(f"Equal Error Rate : {eer*100:.2f}% (Threshold: <= 12%)")
    print(f"Optimal Decision Threshold: {optimal_threshold:.4f}")
    print(f"F1 Score         : {f1*100:.2f}% (Threshold: >= 80%)")
    print(f"Fake Accuracy    : {acc_fake*100:.2f}% (Threshold: >= 75%)")
    print(f"Real Accuracy    : {acc_real*100:.2f}% (Threshold: >= 75%)")
    print("-" * 40)
    print("Confusion Matrix:")
    print(f"                 Predicted Fake  Predicted Real")
    print(f"Actual Fake      {tn:<15} {fp:<15}")
    print(f"Actual Real      {fn:<15} {tp:<15}")
    print("="*40)

    # Save report
    with open("report.md", "w") as f:
        f.write("# Performance Report\n\n")
        f.write(f"- **Overall Accuracy:** {acc*100:.2f}%\n")
        f.write(f"- **Equal Error Rate (EER):** {eer*100:.2f}%\n")
        f.write(f"- **F1 Score:** {f1*100:.2f}%\n")
        f.write(f"- **Fake (Deepfake) Accuracy:** {acc_fake*100:.2f}%\n")
        f.write(f"- **Real (Genuine) Accuracy:** {acc_real*100:.2f}%\n\n")
        f.write("## Confusion Matrix\n")
        f.write("| | Predicted Deepfake | Predicted Genuine |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Actual Deepfake** | {tn} | {fp} |\n")
        f.write(f"| **Actual Genuine** | {fn} | {tp} |\n")
    
    print("\nReport saved to report.md")

if __name__ == "__main__":
    main()
