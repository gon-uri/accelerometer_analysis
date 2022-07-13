#%%
import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import gaussian, filtfilt, butter, find_peaks
import scipy.optimize
import seaborn as sns

sns.set_theme(style="ticks")


#%%
path_name_list = []
path = "/Users/uribarri/Documents/Tremor/app_data/files/"
for filename in glob.glob(os.path.join(path, "*.csv")):
    path_name_list.append(filename)

path_name_list.sort()

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

df = pd.DataFrame(
    list(
        zip(
            patient_number_list,
            experiment_number_list,
            file_type_list,
            file_name_list,
            path_name_list,
        )
    ),
    columns=["Patient", "Session", "Type", "Name", "Path"],
)

df_raw_acc = df[df["Type"] == "RAW_ACC_SENSOR"].copy()
df_raw_acc["Patient"] = df_raw_acc["Patient"].astype("category")
df_raw_acc["Session"] = df_raw_acc["Session"].astype("category")

#%%

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
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
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a butterworth bandpass filter to data and return
    the filtered signal in other vector

    """

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def read_signal(file_path, low_freq=5, high_freq=25):
    raw_acc = pd.read_csv(file_path, engine="python")
    raw_acc["total"] = (
        raw_acc["X (m/s^2)"] ** 2
        + raw_acc["Y (m/s^2)"] ** 2
        + raw_acc["Z (m/s^2)"] ** 2
    ) ** (1 / 2)
    acc = raw_acc["total"]
    signal = acc - np.mean(acc)
    times = (raw_acc["Timestamp (ns)"] - raw_acc["Timestamp (ns)"][0]) / 1000000000
    avg_timestep = np.mean(np.diff(times))
    fs = 1 / (avg_timestep)
    filtered_signal = butter_bandpass_filter(signal, low_freq, high_freq, fs)
    return filtered_signal, avg_timestep, fs


def compute_max_freq(signal, avg_timestep):
    freqs = np.fft.fftfreq(signal.size, avg_timestep)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(signal)) ** 2
    positive_lim = int(len(freqs) / 2)  # +1
    positive_freqs = freqs[:positive_lim]
    positive_ps = ps[:positive_lim]
    peaks, _ = find_peaks(positive_ps, distance=50, threshold=np.max(positive_ps) / 8)
    peak_heights = positive_ps[peaks]
    idx = np.argmax(peak_heights)
    max_peak_idx = peaks[idx]
    max_height = peak_heights[idx]/np.sum(positive_ps)
    max_freq = positive_freqs[max_peak_idx]
    return max_freq, max_height


def compute_session_freq_variability(signal, fs, window_width=3, overlap_ratio=0.5):
    step_length = int(window_width * fs)
    window = gaussian(step_length, int(step_length / 6))
    spectrogram, freqs, bins, im = plt.specgram(
        signal,
        window=window,
        NFFT=len(window),
        Fs=fs,
        noverlap=int(overlap_ratio * step_length),
    )
    mean_value = np.mean(spectrogram)
    idx = np.argmax(spectrogram, axis=0)
    powers = np.max(spectrogram, axis=0)
    max_freqs = freqs[idx]
    mean_max_freqs = np.median(max_freqs)
    std_max_freqs = np.std(max_freqs)
    q75, q25 = np.percentile(max_freqs, [75, 25])
    iqr = q75 - q25
    return mean_max_freqs, std_max_freqs, iqr, mean_value

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(
        ff[np.argmax(Fyy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.0**0.5
    guess_offset = 0
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": np.max(pcov),
        "rawres": (guess, popt, pcov),
    }

#%%
max_freq_list = []
max_height_list = []
mean_max_freqs_list = []
std_max_freqs_list = []
iqr_list = []

all_paths = df_raw_acc["Path"].values

for file_path in all_paths:
    signal, avg_timestep, fs = read_signal(file_path)
    max_freq, max_height = compute_max_freq(signal, avg_timestep)
    mean_max_freqs, std_max_freqs, iqr, mean_value = compute_session_freq_variability(
        signal, fs
    )
    iqr_list.append(iqr)
    max_freq_list.append(max_freq)
    max_height_list.append(max_height)
    mean_max_freqs_list.append(mean_max_freqs)
    std_max_freqs_list.append(std_max_freqs)

# %%
df_raw_acc["Max Freq"] = max_freq_list
df_raw_acc["Max Height"] = max_height_list
df_raw_acc["Median Max Freq"] = mean_max_freqs_list
df_raw_acc["Session Variability (std)"] = std_max_freqs_list
df_raw_acc["Session Variability (iqr)"] = iqr_list

# %%
# g = sns.catplot(x="Patient", y="Max Freq", hue="Session", data=df_raw_acc)

# g = sns.catplot(x="Patient", y="Median Max Freq", hue="Session", data=df_raw_acc)

# g = sns.catplot(
#     x="Patient", y="Session Variability (std)", hue="Session", data=df_raw_acc
# )

# g = sns.catplot(
#     x="Patient", y="Session Variability (std)", hue="Session", data=df_raw_acc
# )

# g = sns.catplot(
#     x="Patient", y="Session Variability (iqr)", hue="Session", data=df_raw_acc
# )

# %%
g = sns.catplot(x="Session", y="Max Freq", hue="Patient", data=df_raw_acc)

g = sns.catplot(x="Session", y="Session Variability (std)", hue="Patient", data=df_raw_acc)

g = sns.catplot(x="Session", y="Max Height", hue="Patient", data=df_raw_acc)

# %% Unfiltered signal

# signal, avg_timestep, fs = read_signal(
#     "/Users/uribarri/Documents/Tremor/app_data/files/Dsotp0604_RAW_ACC_SENSOR.csv",
# )

file_path = "/Users/uribarri/Documents/Tremor/app_data/files/Dsotp0610_RAW_ACC_SENSOR.csv"

raw_acc = pd.read_csv(file_path, engine="python")
raw_acc["total"] = (
    raw_acc["X (m/s^2)"] ** 2
    + raw_acc["Y (m/s^2)"] ** 2
    + raw_acc["Z (m/s^2)"] ** 2
) ** (1 / 2)
acc = raw_acc["total"]
signal = acc - np.mean(acc)
times = (raw_acc["Timestamp (ns)"] - raw_acc["Timestamp (ns)"][0]) / 1000000000
avg_timestep = np.mean(np.diff(times))
fs = 1 / (avg_timestep)

low_freq = 5
high_freq = 25

filtered_signal = butter_bandpass_filter(signal, low_freq, high_freq, fs)

times = np.arange(filtered_signal.size) * avg_timestep
freqs = np.fft.fftfreq(filtered_signal.size, avg_timestep)

margin_filter = int(fs)
plt.plot(times[margin_filter:-margin_filter],filtered_signal[margin_filter:-margin_filter])
plt.xlabel("Time")
plt.ylabel("Accelerometer")
plt.savefig('Images/'+ 'raw_signal' + '.svg', format='svg', dpi=1200)
plt.show()

idx = np.argsort(freqs)
ps = np.abs(np.fft.fft(filtered_signal)) ** 2
positive_lim = int(len(freqs) / 2)
positive_freqs = freqs[:positive_lim]
positive_ps = ps[:positive_lim]
peaks, _ = find_peaks(positive_ps, distance=50, threshold=np.max(positive_ps) / 8)
peak_heights = positive_ps[peaks]
idx = np.argmax(peak_heights)
max_peak_idx = peaks[idx]
max_height = peak_heights[idx]
max_freq = positive_freqs[max_peak_idx]

plt.plot(positive_freqs, positive_ps)
plt.plot(positive_freqs[peaks], positive_ps[peaks], "x")
plt.axvline(x=low_freq, color="r", linestyle="-")
plt.axvline(x=high_freq, color="r", linestyle="-")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0.0, 30)
plt.savefig('Images/'+ 'raw_filter' +'.svg', format='svg', dpi=1200)
plt.show()

# %% One Signal Analysis

idx = np.argsort(freqs)
ps = np.abs(np.fft.fft(filtered_signal)) ** 2
positive_lim = int(len(freqs) / 2)
positive_freqs = freqs[:positive_lim]
positive_ps = ps[:positive_lim]
peaks, _ = find_peaks(positive_ps, distance=50, threshold=np.max(positive_ps) / 8)
peak_heights = positive_ps[peaks]
idx = np.argmax(peak_heights)
max_peak_idx = peaks[idx]
max_height = peak_heights[idx]
max_freq = positive_freqs[max_peak_idx]

upper_freq = max_freq + 1
lower_freq = max_freq - 1

max_freq_filtered_signal = butter_bandpass_filter(signal, lower_freq, upper_freq, fs)

plt.plot(positive_freqs, positive_ps)
plt.plot(positive_freqs[peaks], positive_ps[peaks], "x")
plt.axvline(x=lower_freq, color="r", linestyle="-")
plt.axvline(x=upper_freq, color="r", linestyle="-")
plt.xlabel("Frequency (Hz)")
plt.xlim(0.0, 30)
plt.savefig('Images/' + 'peak_filtered' + '.svg', format='svg', dpi=1200)
plt.show()


# %% Max peak filter zoom

plt.plot(
    times[200+margin_filter:500+margin_filter], signal[200+margin_filter:500+margin_filter], alpha=0.6
)
plt.plot(
    times[200+margin_filter:500+margin_filter],
    max_freq_filtered_signal[200+margin_filter:500+margin_filter],
    alpha=0.6,
)
plt.xlabel("Time (s)")
plt.ylim(-0.15, 0.15)
plt.savefig('Images/' + 'filtered_signal_zoom' + '.svg', format='svg', dpi=1200)
plt.show()
plt.show()


# %% Fitting Curve Sinusoidal

res = fit_sin(times[margin_filter:-margin_filter], max_freq_filtered_signal[margin_filter:-margin_filter])
print(
    "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s"
    % res
)

inicio = margin_filter
final = -margin_filter
fitted_curve = res["fitfunc"](times)

plt.plot(
    times[inicio:final], max_freq_filtered_signal[inicio:final], label="y", linewidth=1
)
plt.plot(
    times[inicio:final], fitted_curve[inicio:final], label="y fit curve", linewidth=1
)
plt.legend(loc="best")
plt.show()



# %% 
window = int(fs/max_freq)
abs_times = times[margin_filter:-margin_filter]
abs_signal = np.abs(max_freq_filtered_signal[margin_filter:-margin_filter])

max_times = []
max_signal = []
for i in range(window,len(abs_signal)-window):
    max_times.append(abs_times[i])
    max_signal.append(np.max(abs_signal[i-window:i+window]))

mean_amp = np.mean(max_signal)
std_amp = np.std(max_signal)

plt.plot(
    max_times, max_signal
)

plt.axhline(y=mean_amp, color="r", linestyle="-")
plt.axhline(y=mean_amp+std_amp, color="r", linestyle="--")
plt.axhline(y=mean_amp-std_amp, color="r", linestyle="--")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.ylim(0, 0.15)
plt.savefig('Images/' + 'Amplitude estimation' + '.svg', format='svg', dpi=1200)
plt.show()
plt.show()


# %% Max peak filter

margin_filter = int(fs)
rms = (
    np.mean(max_freq_filtered_signal[margin_filter:-margin_filter] ** 2) ** (0.5)
) * (np.sqrt(2))

plt.axhline(y=mean_amp, color="r", linestyle="-")
plt.axhline(y=-mean_amp, color="r", linestyle="-")

plt.plot(
    times[margin_filter:-margin_filter], signal[margin_filter:-margin_filter], alpha=0.6
)
plt.plot(
    times[margin_filter:-margin_filter],
    max_freq_filtered_signal[margin_filter:-margin_filter],
    alpha=0.6,
)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.ylim(-0.15, 0.15)
plt.savefig('Images/' + 'filtered_signal' + '.svg', format='svg', dpi=1200)
plt.show()
plt.show()


# %%
print(mean_amp)
print(std_amp)
print(rms)

# %%
def compute_amplitude(signal,max_freq,fs):
    margin_filter = int(fs)
    upper_freq = max_freq + 1
    lower_freq = max_freq - 1

    max_freq_filtered_signal = butter_bandpass_filter(signal, lower_freq, upper_freq, fs)
    window = int(fs/max_freq)

    abs_times = times[margin_filter:-margin_filter]
    abs_signal = np.abs(max_freq_filtered_signal[margin_filter:-margin_filter])

    max_times = []
    max_signal = []
    for i in range(window,len(abs_signal)-window):
        max_times.append(abs_times[i])
        max_signal.append(np.max(abs_signal[i-window:i+window]))

    mean_amp = np.mean(max_signal)
    std_amp = np.std(max_signal)

    return mean_amp, std_amp