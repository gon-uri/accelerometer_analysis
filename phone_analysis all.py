#%%
import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import gaussian,filtfilt,butter,find_peaks
import seaborn as sns
sns.set_theme(style="ticks")

path_name_list = []
path = '/Users/uribarri/Documents/Tremor/app_data/files/'
for filename in glob.glob(os.path.join(path, '*.csv')):
    path_name_list.append(filename)

path_name_list.sort()

#%%
patient_number_list = []
experiment_number_list = []
file_type_list = []
file_name_list = []

for path_string in path_name_list:
    file_name = path_string.split("/")[-1]
    file_name = file_name.split(".")[0]
    file_name_list.append(file_name)
    patient_number_list.append(int(file_name[5:7]))
    experiment_number_list.append(int(file_name[7:9]))
    file_type_list.append(file_name[10:])

df = pd.DataFrame(list(zip(patient_number_list,experiment_number_list,file_type_list,file_name_list,path_name_list)), columns =['Patient', 'Session', 'Type', 'Name','Path'])

df_raw_acc = df[df['Type']=='RAW_ACC_SENSOR'].copy()
df_raw_acc["Patient"] = df_raw_acc["Patient"].astype("category")
df_raw_acc["Session"] = df_raw_acc["Session"].astype("category")

#%%

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Generates the coefficients for a butterworth bandpass filter

    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a butterworth bandpass filter to data and return 
    the filtered signal in other vector 

    """

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def read_signal(file_path, low_freq = 5, high_freq = 25):
    raw_acc = pd.read_csv(file_path, engine='python')
    raw_acc['total'] = (raw_acc['X (m/s^2)']**2 + raw_acc['Y (m/s^2)']**2 + raw_acc['Z (m/s^2)']**2)**(1/2)
    acc = raw_acc['total']
    signal = acc - np.mean(acc)
    times = (raw_acc['Timestamp (ns)'] - raw_acc['Timestamp (ns)'][0])/1000000000
    avg_timestep = np.mean(np.diff(times))
    fs = 1/(avg_timestep)
    filtered_signal = butter_bandpass_filter(signal,low_freq,high_freq,fs)
    return filtered_signal, avg_timestep, fs

def compute_max_freq(signal,avg_timestep):
    freqs = np.fft.fftfreq(signal.size, avg_timestep)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(signal))**2
    positive_lim = int(len(freqs)/2)+1
    positive_freqs = freqs[:positive_lim]
    positive_ps = ps[:positive_lim]
    peaks, _ = find_peaks(positive_ps, distance=50, threshold = np.max(positive_ps)/8)
    peak_heights = positive_ps[peaks]
    idx = np.argmax(peak_heights)
    max_peak_idx = peaks[idx]
    # max_height = peak_heights[idx]/np.sum(positive_ps)
    limite = int(positive_lim/200)
    from_freq = positive_freqs[max_peak_idx-limite]
    to_freq = positive_freqs[max_peak_idx+limite]
    print('from:',from_freq)
    print('to:',to_freq)
    max_height = np.sum(positive_ps[max_peak_idx-limite:max_peak_idx+limite])/np.sum(positive_ps)
    max_freq = positive_freqs[max_peak_idx]
    return max_freq, max_height

def compute_session_freq_variability(signal,fs,window_width=3,overlap_ratio=0.5):
    step_length = int(window_width*fs)
    window = gaussian(step_length,int(step_length/6))
    spectrogram, freqs, bins, im = plt.specgram(signal,window=window, NFFT=len(window), Fs=fs, noverlap=int(overlap_ratio*step_length))
    mean_value = np.mean(spectrogram)
    idx = np.argmax(spectrogram,axis=0)
    powers = np.max(spectrogram,axis=0)
    max_freqs = freqs[idx]
    mean_max_freqs = np.median(max_freqs)
    std_max_freqs = np.std(max_freqs)
    q75, q25 = np.percentile(max_freqs, [75 ,25])
    iqr = q75 - q25
    return mean_max_freqs, std_max_freqs, iqr, mean_value

def compute_amplitude(signal,max_freq,fs):
    margin_filter = int(fs)
    upper_freq = max_freq + 1
    lower_freq = max_freq - 1

    max_freq_filtered_signal = butter_bandpass_filter(signal, lower_freq, upper_freq, fs)
    window = int(fs/max_freq)

    abs_signal = np.abs(max_freq_filtered_signal[margin_filter:-margin_filter])

    max_signal = []
    for i in range(window,len(abs_signal)-window):
        max_signal.append(np.max(abs_signal[i-window:i+window]))

    mean_amp = np.mean(max_signal)
    std_amp = np.std(max_signal)

    return mean_amp, std_amp

#%%
max_freq_list = []
max_height_list = []
mean_max_freqs_list = []
std_max_freqs_list = []
iqr_list = []
amplitude_list = []
std_amp_list = []

all_paths = df_raw_acc['Path'].values

for file_path in all_paths:
    signal, avg_timestep, fs = read_signal(file_path)
    max_freq, max_height = compute_max_freq(signal,avg_timestep)
    mean_max_freqs, std_max_freqs, iqr, mean_value = compute_session_freq_variability(signal,fs)
    mean_amp, std_amp = compute_amplitude(signal,max_freq,fs)

    iqr_list.append(iqr)
    max_freq_list.append(max_freq)
    max_height_list.append(max_height)
    mean_max_freqs_list.append(mean_max_freqs)
    std_max_freqs_list.append(std_max_freqs)
    amplitude_list.append(mean_amp)
    std_amp_list.append(std_amp)


# %%
df_raw_acc['Max Freq'] = max_freq_list
df_raw_acc['Max Height'] = max_height_list
df_raw_acc['Median Max Freq'] = mean_max_freqs_list
df_raw_acc['Session Variability (std)'] = std_max_freqs_list
df_raw_acc['Session Variability (iqr)'] = iqr_list
df_raw_acc['Amplitude'] = amplitude_list
df_raw_acc['Amplitude std'] = std_amp_list

# # %%
# g = sns.catplot(x="Patient", y="Max Freq", hue="Session", data=df_raw_acc)

# # %%
# g = sns.catplot(x="Patient", y="Median Max Freq", hue="Session", data=df_raw_acc)

# # %%
# g = sns.catplot(x="Patient", y="Session Variability (std)", hue="Session", data=df_raw_acc)

# # %%
# g = sns.catplot(x="Patient", y="Session Variability (std)", hue="Session", data=df_raw_acc)

# # %%
# g = sns.catplot(x="Patient", y="Session Variability (iqr)", hue="Session", data=df_raw_acc)

# %%
g = sns.catplot(x="Session", y="Max Freq", hue="Patient", data=df_raw_acc)
plt.savefig('Images/' + 'max_freq' + '.svg', format='svg', dpi=1200)
plt.show()

g = sns.catplot(x="Session", y="Session Variability (iqr)", hue="Patient", data=df_raw_acc)
plt.savefig('Images/' + 'variability' + '.svg', format='svg', dpi=1200)
plt.show()

g = sns.catplot(x="Session", y="Max Height", hue="Patient", data=df_raw_acc)
plt.savefig('Images/' + 'height' + '.svg', format='svg', dpi=1200)
plt.show()

#%%
g = sns.scatterplot(x="Max Freq", y="Max Height", hue="Patient", data=df_raw_acc)
plt.savefig('Images/' + 'freq_vs_height' + '.svg', format='svg', dpi=1200)
plt.ylabel("Peak Power")
plt.savefig('Images/' + 'height' + '.svg', format='svg', dpi=1200)
plt.show()

# %%
g = sns.scatterplot(x="Max Freq", y="Amplitude", hue="Patient", data=df_raw_acc)
plt.ylabel("Amplitude")
plt.xlabel("Max Frequency (Hz)")
plt.xlim(13.5,17.5)
plt.ylim(0,0.35)
plt.savefig('Images/' + 'Amp_vs_freq' + '.svg', format='svg', dpi=1200)
plt.show()


# %%'Amplitude std'
g = sns.scatterplot(x="Amplitude", y="Amplitude std", hue="Patient", data=df_raw_acc)

plt.xlim(0,0.35)
plt.ylim(0,0.25)

plt.savefig('Images/' + 'Amp_vs_std' + '.svg', format='svg', dpi=1200)

plt.show()


# %%'Amplitude std'
g = sns.catplot(x="Session", y="Amplitude std", hue="Patient", data=df_raw_acc)
plt.ylim(0,1.5)

# %% AMPLITUDE

g = sns.catplot(x="Session", y="Amplitude", hue="Patient", data=df_raw_acc)
plt.savefig('Images/' + 'Amp' + '.svg', format='svg', dpi=1200)
plt.show()



# %% AMPLITUDE ZOOM

g = sns.catplot(x="Session", y="Amplitude", hue="Patient", data=df_raw_acc)
plt.ylim(0,0.5)
plt.savefig('Images/' + 'Amp_ZOOM' + '.svg', format='svg', dpi=1200)
plt.show()
# %%

filtered_df = df_raw_acc[df_raw_acc['Session'].isin([1,2,3,4,5,6,8,9,10])]
more_filtered_df = filtered_df[filtered_df['Patient'].isin([6,7,9,15])]

g = sns.lmplot(data=more_filtered_df, x="Max Freq", y="Amplitude", hue="Patient")
plt.ylabel("Vibration Amplitude")
plt.xlabel("Vibration Frequency (Hz)")
plt.savefig('Images/' + 'Amp_regression' + '.jpg', format='jpg', dpi=1200)
plt.show()
# %%
