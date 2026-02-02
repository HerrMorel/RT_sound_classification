##################################################################################################
# This example implements the inference and classification of a pretrained Tensorflow Yamnet model.
# The Yamnet model is used to recognized different kinds of sounds.(refer to the csv file)
# The audio processing will be implemented using Gstreamer. Using a pipeline.
# The audio input can be a microphone or a wav file. Use one of the wav files to test the model.
# The top 5 classifications will be printed for each audio buffer in realtime.
#
# Find and test the microphone:
# On linux:
# arecord -l # Get the list of all microphone
# arecord -D hw:0,0 -f cd -d 5 -r 16000 -c 1 test.wav #Record a test wav 5s, 16000Hz, mono, using card0 device0
# Change hw:0,0 acording to the microphone that is going to be used
# Run this script! The test.wav file will be played and clasified. Use on linux (aplay test.wav) to hear the wav file
##################################################################################################

import tensorflow as tf
import tensorflow_hub as hub
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import csv

#Create the Data pipeline
def build_pipeline():
    pipeline = Gst.parse_launch(
            'filesrc location=test.wav ! wavparse ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,channel=1,rate=16000 ! tee name=t ' #Input a wav file
            #'autoaudiosrc ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,channel=1,rate=16000 ! tee name=t '  #Automatic use one microphone
            #'audiotestsrc freq=440 ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,channel=1,rate=16000 ! tee name=t '  #Test with a 440Hz signal
            #'alsasrc device=hw:0,0 ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,channel=1,rate=16000 ! tee name=t '  #Use this line for realtime classification using microphone input (hw:0,0)
            't. ! queue ! autoaudiosink '      # One output to headphones
            't. ! queue ! appsink name=sink'   # the other output to feed the model
    )
    sink = pipeline.get_by_name("sink")
    sink.set_property("emit-signals",True)
    return pipeline, sink


#Preprocess and Inference, Used on each new data buffer
def handle_pcm(pcm_np):

    # Convert the pcm signal to float32 and normalize(-1,1)
    waveform = pcm_np.astype(np.float32)/32768.0
    #print(waveform.max())
    #print(waveform.min())
    #print(waveform.dtype)
    #print(waveform.shaped)

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


#Get new Data
def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR #ERROR no new data 
    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if ok:
        pcm_np = np.frombuffer(mapinfo.data, dtype=np.int16) 
        buf.unmap(mapinfo)
        handle_pcm(pcm_np) #pass to preprocessing and inference
        return Gst.FlowReturn.OK
    return Gst.FlowReturn.ERROR #ERROR cast to int16 didnt work

#Main LOOP
def start_mainloop(pipeline):
    loop = GLib.MainLoop()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    pipeline.set_state(Gst.State.NULL)



#/////////////////MAIN/////////////////////////

#Init Gstreamer
Gst.init(None)

#Load pretrained YAMNet from tensorflow hub
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

#Build the pipeline
pipeline, sink = build_pipeline()

#Callback function when new sample 
sink.connect("new_sample", on_new_sample)

#Main loop
start_mainloop(pipeline)


