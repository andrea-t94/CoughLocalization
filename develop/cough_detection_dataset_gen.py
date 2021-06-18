# VoiceMed

# file: aug.py
# date of creation: 2021-04-18
# authors: piemonty
# version: 1.0.0
# status: dev
# desc: Average outputs of covid classifier to provide a single probability value
# usage: see __main__ for sample usage

import os
import numpy as np
import tensorflow as tf
import soundfile as sf
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt


HOP_LENGTH = 512  # samples --> time_delta = HOP_LENGTH/frequency (i.e. 0.0106667 s)
N_FTT = 2048      # length of the fft window --> time_delta = N_FTT/frequency (0,0426667 s)

TIME_WINDOW = 0.3  # s , time window to aggregate mfcc columns
HOP_TIME_WINDOW = 0.1  # s , time window to hop when aggregating mfcc columns

OVERLAPPING_TH = 0.8


def get_annotations_in_sec(input_labels_txt):
    with open(input_labels_txt, 'r') as f:
        lines = np.float32([val.split('\t') for val in [line.rstrip() for line in f]])
        original_labels = [item for sublist in lines for item in sublist]
        return original_labels


def is_cough_in_aggregated_mfcc(start_s, stop_s, annotations_s):
    left_annotations = np.array(annotations_s[::2])
    right_annotations = np.array(annotations_s[1::2])
    mfcc_time_interval = stop_s - start_s
    overlapping = False
    overlapping_ratio = 0
    for i in range(len(left_annotations)):
        if start_s > right_annotations[i]:  # mfcc interval is to the right of annotated interval
            continue
        if left_annotations[i] < start_s < right_annotations[i]:  # possible overlapping
            if stop_s > right_annotations[i]:
                overlapping_ratio = (right_annotations[i]-start_s) / mfcc_time_interval
            else:
                overlapping_ratio = 1  # 100%
        elif start_s < left_annotations[i]:  # possible overlapping
            if stop_s > right_annotations[i]:
                overlapping_ratio = (right_annotations[i] - left_annotations[i]) / mfcc_time_interval
            else:
                overlapping_ratio = (stop_s - left_annotations[i]) / mfcc_time_interval
        if overlapping_ratio > OVERLAPPING_TH:
            overlapping = True
            break
    return overlapping


if __name__ == '__main__':
    output_dir = {0: 'coughDetectorTrainSetLabeled/not/', 1: 'coughDetectorTrainSetLabeled/cough/'}
    train_set_dir = 'coughDetectorTrainValSet/train/'
    train_set_subdirs = os.listdir(train_set_dir)
    annotations_paths = []
    wav_paths = []
    unique_hashes = []
    for subdir in train_set_subdirs:
        for file_name in os.listdir(train_set_dir + subdir):
            if file_name.endswith('.wav'):
                wav_paths.append(train_set_dir + subdir + '/' + file_name)
                unique_hashes.append(file_name[:-4])
            elif file_name.endswith('.txt'):
                annotations_paths.append(train_set_dir + subdir + '/' + file_name)

    for file_hash in unique_hashes:
        print(file_hash)
        try:
            single_wav_path = list(filter(lambda x: file_hash in x, wav_paths))[0]
            single_annotation_path = list(filter(lambda x: file_hash in x, annotations_paths))[0]
        except:
            print('could not find all required files for hash: ', file_hash)
            continue
        signal, sr = sf.read(single_wav_path, dtype=np.float32)
        annotations = get_annotations_in_sec(single_annotation_path)
        #plt.plot(np.array(range(len(signal))), signal)
        #plt.show()
        print(file_hash, sr)
        try:
            mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=20)
        except:
            continue

        # display mfccs
        # fig, ax = plt.subplots()
        # img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        # fig.colorbar(img, ax=ax)
        # ax.set(title='MFCC')
        # plt.show()

        number_of_mfcc_cols_to_aggregate = int((TIME_WINDOW*sr - N_FTT)/HOP_LENGTH)
        column_hop = int((HOP_TIME_WINDOW*sr - N_FTT)/HOP_LENGTH)
        for col_start_idx in range(0, mfccs.shape[1] - number_of_mfcc_cols_to_aggregate - 1, column_hop):
            time_start_s = col_start_idx * HOP_LENGTH / sr
            time_stop_s = (col_start_idx+number_of_mfcc_cols_to_aggregate) * HOP_LENGTH / sr
            aggregated_mfcc = mfccs[:, col_start_idx:col_start_idx+number_of_mfcc_cols_to_aggregate]
            does_overlap_cough_burst = is_cough_in_aggregated_mfcc(time_start_s, time_stop_s, annotations)
            if does_overlap_cough_burst:
                np.save(output_dir[does_overlap_cough_burst] + file_hash + '_' + str(col_start_idx) + '.npy',
                        aggregated_mfcc)
