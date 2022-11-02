#%%
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import GccPhat
from speechbrain.processing.multi_mic import DelaySum

import matplotlib.pyplot as plt
import torch

# %%
import glob
file_list = glob.glob("../wave-file-preprocessing/wav_folder/*.wav")
file_list
#%%
fs = 16000
file_path = file_list[6]
print(file_path)
signal = read_audio(file_path)
signal = signal.unsqueeze(0)
#%%
stft = STFT(sample_rate=fs)
cov = Covariance()
gccphat = GccPhat()
delaysum = DelaySum()
istft = ISTFT(sample_rate=fs)

#%%
Xs = stft(signal)
XXs = cov(Xs)
tdoas = gccphat(XXs)
Ys_ds = delaysum(Xs, tdoas)
ys_ds = istft(Ys_ds)
# %%

plt.figure(1)
plt.title('Origin STFT at microphone 1')
plt.imshow(torch.transpose(torch.log(Xs[0,:,:,0,0]**2 + Xs[0,:,:,1,0]**2), 1, 0), origin="lower")
plt.savefig("origin_STFT.png")
plt.figure(2)
plt.title('Origin signal at microphone 1')
plt.savefig("origin_wave.png")
plt.plot(signal.squeeze()[:,0])
plt.figure(3)
plt.title('Beamformed signal')
plt.imshow(torch.transpose(torch.log(Ys_ds[0,:,:,0,0]**2 + Ys_ds[0,:,:,1,0]**2), 1, 0), origin="lower")
plt.savefig("beamformed_STFT.png")
plt.figure(4)
plt.title('Beamformed signal')
plt.plot(ys_ds.squeeze())
plt.savefig("beamformed_wave.png")
plt.show()
# %%
import librosa
import soundfile as sf
# %%
for _file in file_list:
    print(f"_file: {_file}")
    file_name = _file.split("/")[-1][:-4]
    # read audio
    signal = read_audio(_file).unsqueeze(0)
    # denoise
    Xs = stft(signal)
    XXs = cov(Xs)
    tdoas = gccphat(XXs)
    Ys_ds = delaysum(Xs, tdoas)
    ys_ds = istft(Ys_ds)
    # show result image
    plt.figure(1)
    plt.title('Origin STFT at microphone 1')
    plt.imshow(torch.transpose(torch.log(Xs[0,:,:,0,0]**2 + Xs[0,:,:,1,0]**2), 1, 0), origin="lower")
    plt.savefig(f"img/{file_name}-origin_STFT.png")
    plt.figure(2)
    plt.title('Origin signal at microphone 1')
    plt.plot(signal.squeeze()[:,0])
    plt.savefig(f"img/{file_name}-origin_wave.png")
    plt.figure(3)
    plt.title('Beamformed signal')
    plt.imshow(torch.transpose(torch.log(Ys_ds[0,:,:,0,0]**2 + Ys_ds[0,:,:,1,0]**2), 1, 0), origin="lower")
    plt.savefig(f"img/{file_name}-beamformed_STFT.png")
    plt.figure(4)
    plt.title('Beamformed signal')
    plt.plot(ys_ds.squeeze())
    plt.savefig(f"img/{file_name}-beamformed_wave.png")
    plt.show()

    # save audio
    sf.write(f"beamformed/{file_name}_denoise.wav", ys_ds.squeeze().numpy(), fs)
# %%
