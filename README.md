# RT_sound_classification (rt_human_sound_detect.py)
This example demonstrates inference and classification of sounds using a pretrained TensorFlow YAMNet model. 
The YAMNet model is used to recognize different kinds of sounds (refer to the CSV file).
Audio processing is implemented using a GStreamer pipeline.
The audio input can be either a microphone or a WAV file. Use one of the provided WAV files to test the model.
The top 5 classifications will be printed for each audio buffer in real time.

################################
Quickstart (on Ubuntu):
#install python3
sudo apt-get install python3

#install tensorflow
pip install tensorflow
pip install tensorflow_hub

#install GStreamer
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

#Run application
python3 rt_human_sound_detect.py
################################

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


