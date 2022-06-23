# First Analysis

#%%
from re import X
#from matplotlib.lines import _LineStyle
import pandas as pd
amp_spectrum = pd.read_csv('data/Dsotp0600_AMP_SPECTRUM.csv')
pow_spectrum = pd.read_csv('data/Dsotp0600_POW_SPECTRUM.csv')
raw_acc = pd.read_csv('data/Dsotp0600_RAW_ACC_SENSOR.csv')
raw_gyro = pd.read_csv('data/Dsotp0600_RAW_GYRO_SENSOR.csv')
total_acc = pd.read_csv('data/Dsotp0600_TOTAL_ACC.csv')


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# %%
raw_acc['total'] = (raw_acc['X (m/s^2)']**2 + raw_acc['Y (m/s^2)']**2 + raw_acc['Z (m/s^2)']**2)**(1/2)
# %%
raw_acc['total'] == total_acc['Total Acceleration (m/s^2)']
# %%
import numpy as np

acc = raw_acc['total']
acc = acc - np.mean(acc)
times = (raw_acc['Timestamp (ns)'] - raw_acc['Timestamp (ns)'][0])/1000000000
avg_timestep = np.mean(np.diff(times))
fs = 1/(avg_timestep)


# %%
#signal_psd = plt.psd(acc, 512, 1, label='Sum data')
#plt.plot()

# %%
freqs = np.fft.fftfreq(times.size, avg_timestep)
idx = np.argsort(freqs)
ps = np.abs(np.fft.fft(acc))**2

positive_lim = int(len(freqs)/2)+1
positive_freqs = freqs[:positive_lim]
positive_ps = ps[:positive_lim]


plt.figure()
plt.plot(freqs[idx], ps[idx])
plt.title('Power spectrum (np.fft.fft)')
plt.xlabel('Frequency (Hz)')
plt.xlim(0,30)
plt.show()

# %%
from scipy.signal import find_peaks
peaks, _ = find_peaks(positive_ps, distance=50, threshold = np.max(positive_ps)/2)
plt.plot(positive_freqs,positive_ps)
plt.plot(positive_freqs[peaks], positive_ps[peaks], "x")
plt.xlabel('Frequency (Hz)')
plt.xlim(0.0,30)
plt.show()

# %%
plt.plot(times,acc)
plt.xlim(1.0,1.2)

# %%
step_length = int(2*fs)
from matplotlib import gridspec
from scipy import signal

gs = gridspec.GridSpec(2, 2, height_ratios=[2,2],hspace=0.5)
#SCIs_masked = np.ma.masked_where(signal_SCIs==0 , signal_SCIs)
#FUNDs_masked = np.ma.masked_where(signal_fund_freqs==0 , signal_fund_freqs)

ax3 = plt.subplot(gs[2:])#, sharex=ax1)
window = signal.gaussian(step_length,int(step_length/6))
spectrogram, freqs, bins, im = plt.specgram(acc,window=window, NFFT=len(window), Fs=fs, noverlap=int(2*step_length / 4))
#plt.plot(times,FUNDs_masked,c='r')
idx = np.argmax(spectrogram,axis=0)
powers = np.max(spectrogram,axis=0)
max_freqs = freqs[idx]

ax1 = plt.subplot(gs[0])
plt.hist(powers, bins=30,range=[np.min(spectrogram), np.max(spectrogram)])
mean_value = np.mean(spectrogram)
plt.axvline(mean_value, color='red',linestyle='dashed')
ax1.set_title('Peaks Power Distribution')

ax1 = plt.subplot(gs[1])
plt.hist(max_freqs, bins=30,range=[0.0, 50.0])
mean_value = np.mean(spectrogram)
ax1.set_title('Peaks Freq Distribution')

#ax1.set_ylim( 15 , 25.0)

ax3.set_ylim( 0.0 , 50.0)
ax3.set_ylabel('Frequency (hz)')
ax3.set_xlabel('Time (s)')
plt.show()
# %%
