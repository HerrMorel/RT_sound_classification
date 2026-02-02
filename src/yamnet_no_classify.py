
# !!!!!!!!!!!!!!!READ this First!!!!!!!!!!!!!!!!!!!
# This pytorch repo (torch_vggish_yamnet) does not include a classifier head and 
# should not be used for classification of sounds. 
# Use only the embeddings output of the model to fine-tune your classifier.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet import vggish
from torch_vggish_yamnet.input_proc import *
from scipy.io import wavfile
import csv
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms



# Read wav file
in_sr, data =wavfile.read("dog.wav")
data = data/32768.0 #Normalized
#data, in_sr = sf.read("dog.wav", dtype="float32")
#data, in_sr = torchaudio.load("t2.wav", normalize=True) #tensor output
print("Waveform tensor shape:",data.shape)
print("Wav min:",data.min())
print("Wav max:",data.max())

#Plot Wav Signal(doesnt work with tensors)
t = range(len(data))
plt.plot(t,data)
plt.show()


# There are two ways to preprocess the wav signal to feed the model
# 1) Using the preprocessing in torch_vggish_yamnet or 
# 2) Calculate everything manually


# 1)Format wav signal and calculate MelSpec to feed the model
converter = WaveformToInput() #Preproc on torch_vggish_yamnet
data = torch.from_numpy(data).float() # Convert to tensor
data = data.unsqueeze(0) # add 1 dimension
in_tensor = converter(data, in_sr) #Data should be a tensor: [num_audio_channels, num_time_steps]
print("Converted wav to tensor:",in_tensor.shape)

# 2)Format wav signal manually
#x_in = torch.from_numpy(data).float() # Convert to tensor
#x_in = x_in.unsqueeze(0)
#print("wav tensor shape:",x_in.shape)
#print("wav tensor type:",x_in.dtype)
#print("wav sample freq:",in_sr)

# Configure the MelSpec transformation
#MEL = torchaudio.transforms.MelSpectrogram(
#        sample_rate=16000,
#        n_fft=512,
#        win_length=400,
#        hop_length=160,
#        n_mels=64,
#        f_min=125,
#        f_max=7500,
#        window_fn=torch.hann_window,
#        mel_scale="htk",
#        power=2.0,
#        center=False,
#        norm=None,
#        normalized=False
#)

# Calculate the Melspectrum
#in_tensor = MEL(x_in).unsqueeze(0)


# Display properties of the preprocessed wav spectr signal
print("Mel shape:", in_tensor.shape)
print("Mel min:", in_tensor.min().item())
print("Mel max:", in_tensor.max().item())
print("Mel mean:", in_tensor.mean().item())


# Models init
embedding_yamnet = yamnet.yamnet(pretrained=True)

# Embedding (forward)
with torch.no_grad():
    emb_yamnet, _ = embedding_yamnet(in_tensor)  # discard logits. !!Use embeddings.!!!Dont use it for classification!!

print("yamnet sembeddings shape:",emb_yamnet.shape)

