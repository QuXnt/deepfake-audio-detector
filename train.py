import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Ensure memory growth for GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parameters
SAMPLING_RATE = 16000
MAX_DURATION = 3  # seconds
MAX_SAMPLES = SAMPLING_RATE * MAX_DURATION
BATCH_SIZE = 32

def load_and_preprocess_audio(file_path, label):
    # Load audio
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    
    # audio is (length, channels). Squeeze to 1D if mono
    audio = tf.squeeze(audio, axis=-1)
    
    # Pad or truncate to MAX_SAMPLES
    audio_len = tf.shape(audio)[0]
    if audio_len < MAX_SAMPLES:
        padding = [[0, MAX_SAMPLES - audio_len]]
        audio = tf.pad(audio, padding, "CONSTANT")
    else:
        audio = audio[:MAX_SAMPLES]
        
    # Extract Spectrogram
    # Compute STFT with higher resolution
    stft = tf.signal.stft(audio, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)
    
    # Logarithmic compression to prevent exploding gradients
    spectrogram = tf.math.log(spectrogram + 1e-6)
    
    # Add channel dimension
    spectrogram = tf.expand_dims(spectrogram, -1)
    
    return spectrogram, label

def create_dataset(directory, batch_size=32):
    # Get all file paths
    fake_dir = os.path.join(directory, 'fake')
    real_dir = os.path.join(directory, 'real')
    
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.wav')]
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.wav')]
    
    file_paths = fake_files + real_files
    labels = [0] * len(fake_files) + [1] * len(real_files)
    
    # Shuffle
    indices = np.arange(len(file_paths))
    np.random.shuffle(indices)
    file_paths = np.array(file_paths)[indices]
    labels = np.array(labels)[indices]
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(load_and_preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(256, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    train_dir = r"C:\mars ai\datasets\for-norm\training"
    val_dir = r"C:\mars ai\datasets\for-norm\validation"
    
    print("Creating datasets...")
    train_ds = create_dataset(train_dir, BATCH_SIZE)
    val_ds = create_dataset(val_dir, BATCH_SIZE)
    
    # Get input shape
    for spec, label in train_ds.take(1):
        input_shape = spec.shape[1:]
        print(f"Spectrogram shape: {input_shape}")
        break
        
    print("Building model...")
    model = build_model(input_shape)
    model.summary()
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=r"C:\mars ai\ps2\model\best_model.keras",
        monitor='val_accuracy',
        save_best_only=True
    )
    
    os.makedirs(r"C:\mars ai\ps2\model", exist_ok=True)
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping, model_checkpoint]
    )
    print("Training complete!")

if __name__ == "__main__":
    main()
