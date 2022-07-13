#%%
import c3d
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt

path = 'data_lab/DS_OT_P06_app04.c3d'
reader = c3d.Reader(open(path, 'rb'))

list_analog = []
list_points = []
for i, points, analog in reader.read_frames():
    list_analog.append(analog)
    list_points.append(points)
    #print('frame {}: point {}, analog {}'.format(i, points.shape, analog.shape))


# %%
vec_analog = np.asarray(list_analog)
vec_points = np.asarray(list_points)

fs = 300.0

low_freq = 5
high_freq = 30
order=6

analog_label_list = reader.analog_labels
point_label_list = reader.point_labels

# %%
channel_number_M = list(point_label_list).index('telephone:TelM                ')
channel_number_S = list(point_label_list).index('telephone:TelS                ')
channel_number_L = list(point_label_list).index('telephone:TelL                ')
print(point_label_list[channel_number_S])

tel_s_x_relS = vec_points[:,channel_number_L,0] - vec_points[:,channel_number_S,0]
tel_s_y_relS = vec_points[:,channel_number_L,1] - vec_points[:,channel_number_S,0]
tel_s_z_relS = vec_points[:,channel_number_L,2] - vec_points[:,channel_number_S,0]

tel_s_x = vec_points[:,channel_number_L,0]
tel_s_y = vec_points[:,channel_number_L,1]
tel_s_z = vec_points[:,channel_number_L,2]

times = (1/fs) * np.arange(len(tel_s_x))

plt.plot(times,tel_s_x - np.mean(tel_s_x))
plt.plot(times,tel_s_y - np.mean(tel_s_y))
plt.plot(times,tel_s_z - np.mean(tel_s_z))

plt.xlabel('Time(s)')

# %%

#vector_module = (tel_s_x**2+tel_s_y**2+tel_s_z**2)**(0.5)
vector_module = tel_s_x
vector_module = vector_module - np.mean(vector_module)

freqs = np.fft.fftfreq(vector_module.size, 1/fs)
positive_lim = int(len(freqs)/2)+1
idx = np.argsort(freqs)
ps = np.abs(np.fft.fft(vector_module-np.mean(vector_module)))**2

plt.plot(times,vector_module)
plt.xlabel('Time(s)')
plt.show()

plt.plot(times[0:300],vector_module[0:300])
plt.xlabel('Time(s)')

# %%

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=4):
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
    y = lfilter(b, a, data)
    return y

#filtered_vector_module = butter_bandpass_filter(vector_module,low_freq,high_freq,fs)
filtered_vector_module = butter_highpass_filter(vector_module,low_freq,fs,order=order)

plt.plot(times, filtered_vector_module)

#%%
filtered_freqs = np.fft.fftfreq(filtered_vector_module.size, 1/fs)
filtered_idx = np.argsort(freqs)
filtered_ps = np.abs(np.fft.fft(filtered_vector_module))**2

filtered_positive_lim = int(len(freqs)/2)+1
filtered_positive_freqs = freqs[:positive_lim]
filtered_positive_ps = ps[:positive_lim]

plt.figure()
plt.plot(filtered_freqs[idx], filtered_ps[idx])
plt.title('Power spectrum (np.fft.fft)')
plt.xlabel('Frequency (Hz)')
plt.xlim(0,20)
#plt.ylim(0,30000)
plt.show()


# %%

positive_freqs = freqs[:positive_lim]
positive_ps = ps[:positive_lim]

plt.figure()
plt.plot(freqs[idx], ps[idx])
plt.title('Power spectrum (np.fft.fft)')
plt.xlabel('Frequency (Hz)')
plt.xlim(0,20)
#plt.ylim(0,45000)
plt.show()
# %%
