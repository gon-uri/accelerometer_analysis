#%%
import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import gaussian, filtfilt, butter, find_peaks
import c3d
import seaborn as sns

sns.set_theme(style="ticks")

# Parameters
fs = 2400.0
low_freq =10
high_freq = 1000
order = 6

# %%

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
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
    y = filtfilt(b, a, data)
    return y

#%%
patient_number = '06'
path_name_list = []
path = "/Users/uribarri/Documents/Tremor/lab_data/P"+patient_number
for filename in glob.glob(os.path.join(path, "*.c3d")):
    if "app" in filename:
        path_name_list.append(filename)

path_name_list.sort()

#%%
# path_name_list = ['data_lab/DS_OT_P06_app04.c3d']
# path_name_list = ['/Users/uribarri/Documents/Tremor/lab_data/P06/20220509/DS_OT_P06_app04.c3d']

#%%
graph_name_list = []
for j in range(len(path_name_list)):
    path = path_name_list[j]
    reader = c3d.Reader(open(path, "rb"))
    graph_name_list.append(path[56:-4])
    list_analog = []
    list_points = []
    for i, points, analog in reader.read_frames():
        list_analog.append(analog)
        list_points.append(points)
        # print('frame {}: point {}, analog {}'.format(i, points.shape, analog.shape))

    vec_analog_raw = np.asarray(list_analog)
    vec_analog_transpose = vec_analog_raw.transpose((1, 0, 2))
    vec_analog = np.reshape(
        vec_analog_transpose,
        (vec_analog_transpose.shape[0], vec_analog_transpose.shape[1] * 8),
    )

    analog_label_list = reader.analog_labels

    channel_number_IMU9_x = list(analog_label_list).index(
        "IMU Accelerometers.IMU9_ACC_X   "
    )
    channel_number_IMU9_y = list(analog_label_list).index(
        "IMU Accelerometers.IMU9_ACC_Y   "
    )
    channel_number_IMU9_z = list(analog_label_list).index(
        "IMU Accelerometers.IMU9_ACC_Z   "
    )
    channel_number_IMU10_x = list(analog_label_list).index(
        "IMU Accelerometers.IMU10_ACC_X  "
    )
    channel_number_IMU10_y = list(analog_label_list).index(
        "IMU Accelerometers.IMU10_ACC_Y  "
    )
    channel_number_IMU10_z = list(analog_label_list).index(
        "IMU Accelerometers.IMU10_ACC_Z  "
    )
    channel_number_IMU11_x = list(analog_label_list).index(
        "IMU Accelerometers.IMU11_ACC_X  "
    )
    channel_number_IMU11_y = list(analog_label_list).index(
        "IMU Accelerometers.IMU11_ACC_Y  "
    )
    channel_number_IMU11_z = list(analog_label_list).index(
        "IMU Accelerometers.IMU11_ACC_Z  "
    )
    channel_number_IMU12_x = list(analog_label_list).index(
        "IMU Accelerometers.IMU12_ACC_X  "
    )
    channel_number_IMU12_y = list(analog_label_list).index(
        "IMU Accelerometers.IMU12_ACC_Y  "
    )
    channel_number_IMU12_z = list(analog_label_list).index(
        "IMU Accelerometers.IMU12_ACC_Z  "
    )
    channel_number_IMU13_x = list(analog_label_list).index(
        "IMU Accelerometers.IMU13_ACC_X  "
    )
    channel_number_IMU13_y = list(analog_label_list).index(
        "IMU Accelerometers.IMU13_ACC_Y  "
    )
    channel_number_IMU13_z = list(analog_label_list).index(
        "IMU Accelerometers.IMU13_ACC_Z  "
    )
    channel_number_IMU14_x = list(analog_label_list).index(
        "IMU Accelerometers.IMU14_ACC_X  "
    )
    channel_number_IMU14_y = list(analog_label_list).index(
        "IMU Accelerometers.IMU14_ACC_Y  "
    )
    channel_number_IMU14_z = list(analog_label_list).index(
        "IMU Accelerometers.IMU14_ACC_Z  "
    )
    channel_number_IMU15_x = list(analog_label_list).index(
        "IMU Accelerometers.IMU15_ACC_X  "
    )
    channel_number_IMU15_y = list(analog_label_list).index(
        "IMU Accelerometers.IMU15_ACC_Y  "
    )
    channel_number_IMU15_z = list(analog_label_list).index(
        "IMU Accelerometers.IMU15_ACC_Z  "
    )

    channel_IMU9_x = vec_analog[channel_number_IMU9_x, :]
    channel_IMU9_y = vec_analog[channel_number_IMU9_y, :]
    channel_IMU9_z = vec_analog[channel_number_IMU9_z, :]

    channel_IMU10_x = vec_analog[channel_number_IMU10_x, :]
    channel_IMU10_y = vec_analog[channel_number_IMU10_y, :]
    channel_IMU10_z = vec_analog[channel_number_IMU10_z, :]

    channel_IMU11_x = vec_analog[channel_number_IMU11_x, :]
    channel_IMU11_y = vec_analog[channel_number_IMU11_y, :]
    channel_IMU11_z = vec_analog[channel_number_IMU11_z, :]

    channel_IMU12_x = vec_analog[channel_number_IMU12_x, :]
    channel_IMU12_y = vec_analog[channel_number_IMU12_y, :]
    channel_IMU12_z = vec_analog[channel_number_IMU12_z, :]

    channel_IMU13_x = vec_analog[channel_number_IMU13_x, :]
    channel_IMU13_y = vec_analog[channel_number_IMU13_y, :]
    channel_IMU13_z = vec_analog[channel_number_IMU13_z, :]

    channel_IMU14_x = vec_analog[channel_number_IMU14_x, :]
    channel_IMU14_y = vec_analog[channel_number_IMU14_y, :]
    channel_IMU14_z = vec_analog[channel_number_IMU14_z, :]

    channel_IMU15_x = vec_analog[channel_number_IMU15_x, :]
    channel_IMU15_y = vec_analog[channel_number_IMU15_y, :]
    channel_IMU15_z = vec_analog[channel_number_IMU15_z, :]

    first_channels_list = [
        channel_IMU9_x,
        channel_IMU9_y,
        channel_IMU9_z,
        channel_IMU10_x,
        channel_IMU10_y,
        channel_IMU10_z,
        channel_IMU11_x,
        channel_IMU11_y,
        channel_IMU11_z,
        channel_IMU12_x,
        channel_IMU12_y,
        channel_IMU12_z,
    ]

    first_channels_list_names = [
        "IMU9_x",
        "IMU9_y",
        "IMU9_z",
        "IMU10_x",
        "IMU10_y",
        "IMU10_z",
        "IMU11_x",
        "IMU11_y",
        "IMU11_z",
        "IMU12_x",
        "IMU12_y",
        "IMU12_z",
    ]

    second_channels_list = [
        channel_IMU13_x,
        channel_IMU13_y,
        channel_IMU13_z,
        channel_IMU14_x,
        channel_IMU14_y,
        channel_IMU14_z,
        channel_IMU15_x,
        channel_IMU15_y,
        channel_IMU15_z,
    ]
    
    second_channels_list_names = [
        "IMU13_x",
        "IMU13_y",
        "IMU13_z",
        "IMU14_x",
        "IMU14_y",
        "IMU14_z",
        "IMU15_x",
        "IMU15_y",
        "IMU15_z",
    ]

    channel_IMU9 = (
        channel_IMU9_x**2 + channel_IMU9_y**2 + channel_IMU9_z**2
    ) ** (1 / 2)
    channel_IMU10 = (
        channel_IMU10_x**2 + channel_IMU10_y**2 + channel_IMU10_z**2
    ) ** (1 / 2)
    channel_IMU11 = (
        channel_IMU11_x**2 + channel_IMU11_y**2 + channel_IMU11_z**2
    ) ** (1 / 2)
    channel_IMU12 = (
        channel_IMU12_x**2 + channel_IMU12_y**2 + channel_IMU12_z**2
    ) ** (1 / 2)
    channel_IMU13 = (
        channel_IMU13_x**2 + channel_IMU13_y**2 + channel_IMU13_z**2
    ) ** (1 / 2)
    channel_IMU14 = (
        channel_IMU14_x**2 + channel_IMU14_y**2 + channel_IMU14_z**2
    ) ** (1 / 2)
    channel_IMU15 = (
        channel_IMU15_x**2 + channel_IMU15_y**2 + channel_IMU15_z**2
    ) ** (1 / 2)

    module_channels_list = [
        channel_IMU9,
        channel_IMU10,
        channel_IMU11,
        channel_IMU12,
        channel_IMU13,
        channel_IMU14,
        channel_IMU15,
    ]
    module_channels_list_names = [
        "IMU9",
        "IMU10",
        "IMU11",
        "IMU12",
        "IMU13",
        "IMU14",
        "IMU15",
    ]

    # for i in range(len(first_channels_list)):

    #     vector_module = first_channels_list[i]
    #     vector_module = vector_module - np.mean(vector_module)

    #     filtered_vector_module = butter_highpass_filter(vector_module,low_freq,fs,order=order)

    #     times = (1/fs) * np.arange(len(filtered_vector_module))

    #     filtered_freqs = np.fft.fftfreq(filtered_vector_module.size, 1/fs)
    #     filtered_idx = np.argsort(filtered_freqs)
    #     filtered_ps = np.abs(np.fft.fft(filtered_vector_module))**2

    #     filtered_positive_lim = int(len(filtered_freqs)/2)+1
    #     filtered_positive_freqs = filtered_freqs[:filtered_positive_lim]
    #     filtered_positive_ps = filtered_ps[:filtered_positive_lim]
    #     filtered_positive_ps = filtered_positive_ps/np.max(filtered_positive_ps)

    #     plt.plot(filtered_positive_freqs,filtered_positive_ps,label = first_channels_list_names[i],alpha=0.3)
    # #plt.legend()
    # plt.title(graph_name_list[j]+' - Power spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.xlim(0,20)
    # #plt.ylim(0,10000)
    # #plt.savefig('Images/'+ graph_name_list[j] + '_phone' + '.svg', format='svg', dpi=1200)
    # plt.show()

    # for i in range(len(second_channels_list)):

    #     vector_module = second_channels_list[i]
    #     vector_module = vector_module - np.mean(vector_module)

    #     filtered_vector_module = butter_highpass_filter(vector_module,low_freq,fs,order=order)

    #     times = (1/fs) * np.arange(len(filtered_vector_module))

    #     filtered_freqs = np.fft.fftfreq(filtered_vector_module.size, 1/fs)
    #     filtered_idx = np.argsort(filtered_freqs)
    #     filtered_ps = np.abs(np.fft.fft(filtered_vector_module))**2

    #     filtered_positive_lim = int(len(filtered_freqs)/2)+1
    #     filtered_positive_freqs = filtered_freqs[:filtered_positive_lim]
    #     filtered_positive_ps = filtered_ps[:filtered_positive_lim]
    #     filtered_positive_ps = filtered_positive_ps/np.max(filtered_positive_ps)

    #     plt.plot(filtered_positive_freqs,filtered_positive_ps,label = second_channels_list_names[i],alpha=0.3)
    # #plt.legend()
    # plt.title(graph_name_list[j]+' - Power spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.xlim(0,20)
    # #plt.ylim(0,10000)
    # #plt.savefig('Images/'+ graph_name_list[j] + '.svg', format='svg', dpi=1200)
    # plt.show()

    for i in range(len(module_channels_list)):

        vector_module = module_channels_list[i]
        vector_module = vector_module - np.mean(vector_module)

        #filtered_vector_module = butter_highpass_filter(
        #    vector_module, low_freq, fs, order=order
        #)

        filtered_vector_module = butter_bandpass_filter(vector_module,low_freq,high_freq,fs,order=order)

        times = (1 / fs) * np.arange(len(filtered_vector_module))

        filtered_freqs = np.fft.fftfreq(filtered_vector_module.size, 1 / fs)
        filtered_idx = np.argsort(filtered_freqs)
        filtered_ps = np.abs(np.fft.fft(filtered_vector_module)) ** 2

        filtered_positive_lim = int(len(filtered_freqs) / 2) + 1
        filtered_positive_freqs = filtered_freqs[:filtered_positive_lim]
        filtered_positive_ps = filtered_ps[:filtered_positive_lim]
        #filtered_positive_ps = filtered_positive_ps / np.max(filtered_positive_ps)

        plt.plot(
            filtered_positive_freqs,
            filtered_positive_ps,
            label=module_channels_list_names[i],
            alpha=0.3
        )
    # plt.legend()
    plt.title('patient'+patient_number+ '_' + graph_name_list[j] + " - Power spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 25)
    plt.legend()
    # plt.ylim(0,10000)
    plt.savefig('Images_IMU/'+ 'patient'+patient_number+ '_' + graph_name_list[j] + '.svg', format='svg', dpi=1200)
    plt.show()
