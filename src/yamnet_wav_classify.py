import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import numpy as np
import csv 

# Load audio
waveform, sr = sf.read("catandwoman.wav")
print("Wav sample freq = ",sr)

# Convert to mono if needed
if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)

# YAMNet expects float32
waveform = waveform.astype(np.float32)

# Load YAMNet from TF Hub
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Run inference
scores, embeddings, spectrogram = yamnet_model(waveform)

# Average scores over time frames
mean_scores = tf.reduce_mean(scores, axis=0)

# Get top-5 predictions
top_values, top_indices = tf.math.top_k(mean_scores, k=5)

# Convert to numpy for printing
top_values = top_values.numpy()
top_indices = top_indices.numpy()+1 #Because the first row on csv file are titles

# Read the CSV class file
with open("yamnet_class_map.csv") as f:
     reader = csv.reader(f)
     labels = [row[2] for row in reader]

# Display the top 5 Classes
for i in range(5):
    print(f"{i+1}. {labels[top_indices[i]]}: {top_values[i]}")

