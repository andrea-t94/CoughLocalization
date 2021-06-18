import numpy as np
import tensorflow as tf
import keras
import soundfile as sf
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from cough_detection_dataset_gen import OVERLAPPING_TH
from cough_detection_dataset_gen import HOP_LENGTH, N_FTT, TIME_WINDOW, HOP_TIME_WINDOW

#model = tf.keras.models.load_model('cough_detector/lm_cough_detector_TH' + str(int(OVERLAPPING_TH*100)) + '.h5')
model = tf.keras.models.load_model('cough_detector/lm_cough_detector_TH80_w300.h5')
test_set_dir = 'coughDetectorTrainValSet/test/'
test_set_subdirs = os.listdir(test_set_dir)

wav_paths = []
for subdir in test_set_subdirs:
    for file_name in os.listdir(test_set_dir + subdir):
        if file_name.endswith('.wav'):
            wav_paths.append(test_set_dir + subdir + '/' + file_name)

for file_path in wav_paths[:10]:
    print(file_path)
    signal, sr = sf.read(file_path, dtype=np.float32)
    try:
        mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=20)
    except:
        continue

    number_of_mfcc_cols_to_aggregate = int((TIME_WINDOW*sr - N_FTT)/HOP_LENGTH)
    column_hop = int((HOP_TIME_WINDOW*sr - N_FTT)/HOP_LENGTH)
    cough_bursts_intervals = [0]
    for col_start_idx in range(0, mfccs.shape[1] - number_of_mfcc_cols_to_aggregate - 1, column_hop):
        time_start_s = col_start_idx * HOP_LENGTH / sr
        time_stop_s = (col_start_idx+number_of_mfcc_cols_to_aggregate) * HOP_LENGTH / sr
        aggregated_mfcc = mfccs[:, col_start_idx:col_start_idx+number_of_mfcc_cols_to_aggregate]
        prediction = model.predict(np.expand_dims(aggregated_mfcc, axis=0))
        if prediction > 0.5:
            if time_start_s>cough_bursts_intervals[-1]:
                cough_bursts_intervals.append(time_start_s)
                cough_bursts_intervals.append(time_stop_s)
            else:
                cough_bursts_intervals[-1] = time_stop_s
    print(cough_bursts_intervals)
    plt.plot(np.array(range(len(signal)))/48000, signal)
    for vertical in cough_bursts_intervals[1:]:
        plt.axvline(x=vertical, color='red')
    plt.show()
