# First Analysis

#%%
from re import X
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
times = (raw_acc['Timestamp (ns)'] - raw_acc['Timestamp (ns)'][0])/1000000
avg_timestep = np.mean(np.diff(times))
fs = 1/(avg_timestep)


# %%
signal_psd = plt.psd(acc, 512, 1, label='Sum data')
plt.plot()

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
plt.xlim(0,0.03)
plt.show()

# %%
from scipy.signal import find_peaks
peaks, _ = find_peaks(positive_ps, distance=150,threshold=np.mean(positive_ps))
plt.plot(positive_freqs,positive_ps)
plt.plot(positive_freqs[peaks], positive_ps[peaks], "x")
plt.show()

# %%
