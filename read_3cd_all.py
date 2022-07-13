#%%
import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import gaussian,filtfilt,butter,find_peaks
import c3d
import seaborn as sns
sns.set_theme(style="ticks")

# Parameters
fs = 300.0
low_freq =10
high_freq = 20
order = 6

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
    y = filtfilt(b, a, data)
    return y

#%%
path_name_list = []
patient_number = '14'
path = '/Users/uribarri/Documents/Tremor/lab_data/P'+patient_number
for filename in glob.glob(os.path.join(path, '*.c3d')):
    if 'app' in filename:
        path_name_list.append(filename)

path_name_list.sort()

#%%
path_name_list = [
'/Users/uribarri/Documents/Tremor/lab_data/P14/DS_OT_P14_app01.c3d', 
#'/Users/uribarri/Documents/Tremor/lab_data/P14/DS_OT_P14_app02.c3d',
'/Users/uribarri/Documents/Tremor/lab_data/P14/DS_OT_P14_app03.c3d',
'/Users/uribarri/Documents/Tremor/lab_data/P14/DS_OT_P14_app04.c3d',
'/Users/uribarri/Documents/Tremor/lab_data/P14/DS_OT_P14_app05.c3d',
'/Users/uribarri/Documents/Tremor/lab_data/P14/DS_OT_P14_app06.c3d'
]

#%%
graph_name_list = []
for j in range(len(path_name_list)):
    path = path_name_list[j]
    reader = c3d.Reader(open(path, 'rb'))
    graph_name_list.append(path[56:-4])
    list_analog = []
    list_points = []
    for i, points, analog in reader.read_frames():
        list_analog.append(analog)
        list_points.append(points)
        #print('frame {}: point {}, analog {}'.format(i, points.shape, analog.shape))

    vec_analog = np.asarray(list_analog)
    vec_points = np.asarray(list_points)

    analog_label_list = reader.analog_labels
    point_label_list = reader.point_labels

    ## PHONE CHANNELS

    channel_number_M = list(point_label_list).index('telephone:TelM                ')
    channel_number_S = list(point_label_list).index('telephone:TelS                ')
    channel_number_L = list(point_label_list).index('telephone:TelL                ')
    
    channel_L_x = vec_points[:,channel_number_L,0]
    channel_L_y = vec_points[:,channel_number_L,1]
    channel_L_z = vec_points[:,channel_number_L,2]

    channel_S_x = vec_points[:,channel_number_S,0]
    channel_S_y = vec_points[:,channel_number_S,1]
    channel_S_z = vec_points[:,channel_number_S,2]

    channel_M_x = vec_points[:,channel_number_M,0]
    channel_M_y = vec_points[:,channel_number_M,1]
    channel_M_z = vec_points[:,channel_number_M,2]

    phone_channels_list = [channel_L_x,channel_L_y,channel_L_z,channel_S_x,channel_S_y,channel_S_z,channel_M_x,channel_M_y,channel_M_z]
    phone_channels_list_names = ['Tel_L_x','Tel_L_y','Tel_L_z','Tel_S_x','Tel_S_y','Tel_S_z','Tel_M_x','Tel_M_y','Tel_M_z']

    for i in range(len(phone_channels_list)):

        vector_module = phone_channels_list[i]
        vector_module = vector_module - np.mean(vector_module)

        #filtered_vector_module = butter_highpass_filter(vector_module,low_freq,fs,order=order)
        filtered_vector_module = butter_bandpass_filter(vector_module,low_freq,high_freq,fs,order=order)

        times = (1/fs) * np.arange(len(filtered_vector_module))

        filtered_freqs = np.fft.fftfreq(filtered_vector_module.size, 1/fs)
        filtered_idx = np.argsort(filtered_freqs)
        filtered_ps = np.abs(np.fft.fft(filtered_vector_module))**2

        filtered_positive_lim = int(len(filtered_freqs)/2)+1
        filtered_positive_freqs = filtered_freqs[:filtered_positive_lim]
        filtered_positive_ps = filtered_ps[:filtered_positive_lim]
        filtered_positive_ps = filtered_positive_ps/np.max(filtered_positive_ps)

        plt.plot(filtered_positive_freqs,filtered_positive_ps,label = phone_channels_list_names[i],alpha=0.3)
    #plt.legend()
    plt.title('patient'+ patient_number +'_'+ graph_name_list[j]+ '_phone' +' - Power spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0,30)
    #plt.ylim(0,10000)
    plt.savefig('Images/'+ 'patient'+ patient_number +'_'+ graph_name_list[j] + '_phone' + '.svg', format='svg', dpi=1200)
    plt.show()

    ## OTHER CHANNELS

    channel_number_LIMU = list(point_label_list).index('DS_OT_P'+patient_number+':LIMU                ')
    channel_number_RIMU = list(point_label_list).index('DS_OT_P'+patient_number+':RIMU                ')
    channel_number_LTIB = list(point_label_list).index('DS_OT_P'+patient_number+':LTIB                ')
    channel_number_LTHAD = list(point_label_list).index('DS_OT_P'+patient_number+':LTHAD               ')
    channel_number_LKNE = list(point_label_list).index('DS_OT_P'+patient_number+':LKNE                ')
    channel_number_LTHI = list(point_label_list).index('DS_OT_P'+patient_number+':LTHI                ')
    channel_number_RPel = list(point_label_list).index('DS_OT_P'+patient_number+':RPel                ')
    channel_number_LTHAP = list(point_label_list).index('DS_OT_P'+patient_number+':LTHAP               ')

    channel_LIMU_x = vec_points[:,channel_number_LIMU,0]
    channel_LIMU_y = vec_points[:,channel_number_LIMU,1]
    channel_LIMU_z = vec_points[:,channel_number_LIMU,2]

    channel_RIMU_x = vec_points[:,channel_number_RIMU,0]
    channel_RIMU_y = vec_points[:,channel_number_RIMU,1]
    channel_RIMU_z = vec_points[:,channel_number_RIMU,2]

    channel_LTIB_x = vec_points[:,channel_number_LTIB,0]
    channel_LTIB_y = vec_points[:,channel_number_LTIB,1]
    channel_LTIB_z = vec_points[:,channel_number_LTIB,2]

    channel_LTHAD_x = vec_points[:,channel_number_LTHAD,0]
    channel_LTHAD_y = vec_points[:,channel_number_LTHAD,1]
    channel_LTHAD_z = vec_points[:,channel_number_LTHAD,2]

    channel_LKNE_x = vec_points[:,channel_number_LKNE,0]
    channel_LKNE_y = vec_points[:,channel_number_LKNE,1]
    channel_LKNE_z = vec_points[:,channel_number_LKNE,2]

    channel_LTHI_x = vec_points[:,channel_number_LTHI,0]
    channel_LTHI_y = vec_points[:,channel_number_LTHI,1]
    channel_LTHI_z = vec_points[:,channel_number_LTHI,2]

    channel_RPel_x = vec_points[:,channel_number_RPel,0]
    channel_RPel_y = vec_points[:,channel_number_RPel,1]
    channel_RPel_z = vec_points[:,channel_number_RPel,2]

    channel_LTHAP_x = vec_points[:,channel_number_LTHAP,0]
    channel_LTHAP_y = vec_points[:,channel_number_LTHAP,1]
    channel_LTHAP_z = vec_points[:,channel_number_LTHAP,2]


    channels_list= [channel_LIMU_x,channel_LIMU_y,channel_LIMU_z,channel_RIMU_x,channel_RIMU_y,channel_RIMU_z,channel_LTIB_x,channel_LTIB_y,channel_LTIB_z,channel_LTHAD_x,channel_LTHAD_y,channel_LTHAD_z]
    channels_list_names= ['LIMU_x','LIMU_y','LIMU_z','RIMU_x','RIMU_y','RIMU_z','LTIB_x','LTIB_y','LTIB_z','LTHAD_x','LTHAD_y','LTHAD_z']

    second_channels_list= [channel_LKNE_x,channel_LKNE_y,channel_LKNE_z,channel_LTHI_x,channel_LTHI_y,channel_LTHI_z,channel_RPel_x,channel_RPel_y,channel_RPel_z,channel_LTHAP_x,channel_LTHAP_y,channel_LTHAP_z]
    second_channels_list_names= ['LKNE_x','LKNE_y','LKNE_z','LTHI_x','LTHI_y','LTHI_z','RPel_x','RPel_y','RPel_z','LTHAP_x','LTHAP_y','LTHAP_z']


    for i in range(len(channels_list)):

        vector_module = channels_list[i]
        vector_module = vector_module - np.mean(vector_module)

        #filtered_vector_module = butter_highpass_filter(vector_module,low_freq,fs,order=order)
        filtered_vector_module = butter_bandpass_filter(vector_module,low_freq,high_freq,fs,order=order)

        times = (1/fs) * np.arange(len(filtered_vector_module))

        filtered_freqs = np.fft.fftfreq(filtered_vector_module.size, 1/fs)
        filtered_idx = np.argsort(filtered_freqs)
        filtered_ps = np.abs(np.fft.fft(filtered_vector_module))**2

        filtered_positive_lim = int(len(filtered_freqs)/2)+1
        filtered_positive_freqs = filtered_freqs[:filtered_positive_lim]
        filtered_positive_ps = filtered_ps[:filtered_positive_lim]
        filtered_positive_ps = filtered_positive_ps/np.max(filtered_positive_ps)

        plt.plot(filtered_positive_freqs,filtered_positive_ps,label = channels_list_names[i],alpha=0.3)
    #plt.legend()
    plt.title('patient'+ patient_number +'_'+ graph_name_list[j]+' - Power spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0,30)
    #plt.ylim(0,10000)
    plt.savefig('Images/'+ 'patient'+ patient_number +'_'+ graph_name_list[j] + '.svg', format='svg', dpi=1200)
    plt.show()

    for i in range(len(second_channels_list)):

        vector_module = second_channels_list[i]
        vector_module = vector_module - np.mean(vector_module)

        #filtered_vector_module = butter_highpass_filter(vector_module,low_freq,fs,order=order)
        filtered_vector_module = butter_bandpass_filter(vector_module,low_freq,high_freq,fs,order=order)


        times = (1/fs) * np.arange(len(filtered_vector_module))

        filtered_freqs = np.fft.fftfreq(filtered_vector_module.size, 1/fs)
        filtered_idx = np.argsort(filtered_freqs)
        filtered_ps = np.abs(np.fft.fft(filtered_vector_module))**2

        filtered_positive_lim = int(len(filtered_freqs)/2)+1
        filtered_positive_freqs = filtered_freqs[:filtered_positive_lim]
        filtered_positive_ps = filtered_ps[:filtered_positive_lim]
        filtered_positive_ps = filtered_positive_ps/np.max(filtered_positive_ps)

        plt.plot(filtered_positive_freqs,filtered_positive_ps,label = second_channels_list_names[i],alpha=0.3)
    #plt.legend()
    plt.title('patient'+ patient_number +'_'+ graph_name_list[j]+  '_second' +' - Power spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0,30)
    #plt.ylim(0,10000)
    plt.savefig('Images/'+ 'patient'+ patient_number +'_'+ graph_name_list[j] +  '_second' + '.svg', format='svg', dpi=1200)
    plt.show()
# %%
