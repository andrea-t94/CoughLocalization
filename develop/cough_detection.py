import numpy as np
import tensorflow as tf
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from models import cnn
import random

mfccs_with_cough_dir = 'coughDetectorTrainSetLabeled/cough/'
mfccs_without_cough_dir = 'coughDetectorTrainSetLabeled/not/'

mfccs_with_cough_paths = [mfccs_with_cough_dir + path for path in os.listdir(mfccs_with_cough_dir)]
mfccs_without_cough_paths = [mfccs_without_cough_dir + path for path in os.listdir(mfccs_without_cough_dir)]\
    [:len(mfccs_with_cough_paths)]  # to retain a balanced dataset

samples = []
labels = []
for file in mfccs_with_cough_paths:
    samples.append(np.load(file))
    labels.append(1)
for file in mfccs_without_cough_paths:
    samples.append(np.load(file))
    labels.append(0)

dataset = list(zip(samples, labels))
random.shuffle(dataset)
samples, labels = zip(*dataset)
del dataset

x = np.array(samples)
y = np.array(labels)
del samples, labels

x_train, x_val = x[:int(0.8*len(x))], x[int(0.8*len(x)):]
y_train, y_val = y[:int(0.8*len(y))], y[int(0.8*len(y)):]
del x, y

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


def get_model(input_shape):
    return cnn(input_shape=input_shape)

if __name__ == '__main__':
    model = get_model(input_shape=(x_train.shape[1], x_train.shape[2], 1))
    #model.summary()
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='binary_crossentropy')
    model.compile(optimizer='adam', loss=bce, metrics='accuracy')
    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=64, epochs=10, verbose=2)
    #model.save('cough_detector/lm_cough_detector_TH' + str(int(OVERLAPPING_TH*100)) + '.h5')
