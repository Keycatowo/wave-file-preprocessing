#%%
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import STFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import Music

import torch
# %%
mics = torch.zeros((4,3), dtype=torch.float)
mics[0,:] = torch.FloatTensor([+0.02, +0.00, +0.00])
mics[1,:] = torch.FloatTensor([+0.06, +0.00, +0.00])
mics[2,:] = torch.FloatTensor([-0.06, +0.00, +0.00])
mics[3,:] = torch.FloatTensor([-0.02, +0.00, +0.00])
# %%
stft = STFT(sample_rate=16000)
cov = Covariance()
music = Music(mics=mics)
# %%
import glob
file_list = glob.glob("../wave-file-preprocessing/wav_folder/*.wav")
file_list
#%%
file_path = file_list[9]
file_path
#%%

# Read the audio file
# signal = read_audio('../wave-file-preprocessing/wav_folder/4-左前.wav')
signal = read_audio(file_path)
signal = signal.unsqueeze(0)
# signal = torch.transpose(signal,0,1)
signal.shape
# %%
Xs = stft(signal)
Xs.shape

XXs = cov(Xs)
XXs.shape
doas = music(XXs)

print(doas)
# %%
