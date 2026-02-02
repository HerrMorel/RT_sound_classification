# RT_sound_classification (rt_human_sound_detect.py)
This example demonstrates inference and classification of sounds using a pretrained TensorFlow YAMNet model. 
The YAMNet model is used to recognize different kinds of sounds (refer to the CSV file).
Audio processing is implemented using a GStreamer pipeline.
The audio input can be either a microphone or a WAV file. Use one of the provided WAV files to test the model.
The top 5 classifications will be printed for each audio buffer in real time.


This project uses third‑party components under the following licenses:
- TensorFlow – Apache License 2.0
- YAMNet (TensorFlow Model Garden) – Apache License 2.0
- GStreamer Core – LGPL 2.1
- GStreamer Plugins (as used) – LGPL 2.1 or other licenses depending on plugin
- NumPy – BSD License
- SciPy – BSD License
- torchaudio – BSD-3-Clause
- matplotlib – Matplotlib License (PSF + BSD)
- Python csv module – PSF License
- SoundFile – BSD-3-Clause
- libsndfile (dependency of SoundFile) – LGPL 2.1


