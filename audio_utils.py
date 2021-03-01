""" REPOSITORY OF UTILS FUNCTIONS """
""" COMPRISING OF FEATURE EXTRACTIONS FUNCTIONS BASED ON LIBROSA OR TENSORFLOW"""

import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import skimage.io
from datetime import date, datetime
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img
import params as spectro_params
import tensorflow as tf
import soundfile as sf
import resampy
import params as spectro_params
import struct




######################################################################
#
# ENVIRONMENTAL VARIABLES
#
######################################################################

warnings.filterwarnings("ignore")
SR = 16000
stft_window_seconds: float = 0.025
stft_hop_seconds: float = 0.010
N_FFT =  int(np.ceil(SR * stft_window_seconds))
HOP_LENGTH = int(np.ceil(SR * stft_hop_seconds))
N_MELS = 64
SILENCE = 0.0018
SAMPLE_LENGTH = 0.96  # s
SAMPLE_SIZE = int(np.ceil(SR * SAMPLE_LENGTH))
NOISE_RATIO = 0.3
PAD = int(np.ceil(SR * SAMPLE_LENGTH* 0.5)) #50% of signal lenght overlapping
N_MFCC = 14
AUGMENT = "extracted-data/Noises/"

params = spectro_params.Params()

######################################################################
#
# FEATURE EXTRACTION FUNCTIONS
#
######################################################################

def envelope(signal, rate, thresh):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    # Create aggregated mean
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for m in y_mean:
        mask.append(m > thresh)

    return mask


def load_audio(path, params):
    signal, rate = librosa.load(path, sr=params.sample_rate)
    mask = envelope(signal, rate, SILENCE)
    signal = signal[mask]

    return signal


def mm_norm(feat):
    scaler = MinMaxScaler()
    scaler.fit(feat)
    return scaler.transform(feat)


def std_norm(feat):
    scaler = StandardScaler()
    scaler.fit(feat)
    return scaler.transform(feat)


def melspectrogram(signal, params):
    signal = librosa.util.normalize(signal)
    spectro = librosa.feature.melspectrogram(
        signal,
        sr=params.sample_rate,
        n_mels=params.mel_bands,
        n_fft=params.n_fft,
        hop_length= params.hop_length
    )
    spectro = librosa.power_to_db(spectro**2, ref=np.max)
    spectro = spectro.astype(np.float32)
    return spectro


def MFCC(sample, params):
    return librosa.feature.mfcc(sample, sr=params.sample_rate, n_mfcc=params.n_mfcc, n_fft=params.n_fft, hop_length=params.hop_length).T



def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Scale to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    V = np.flip(V, axis=0)  # put low frequencies at the bottom in image
    V = 255 - V
    return V

def open_fat_image(img)->Image:
    # open
    x = Image.fromarray(img).convert('RGB')
    # crop
    time_dim, base_dim = x.size
    #crop_x = random.randint(0, time_dim - base_dim)
    #x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])
    # standardize
    img = array_to_img(x, dtype=np.float32)
    return img

def spectrogram_image(signal, params, out, mono=True):
    '''based on librosa, convert audio in logmel-spectrogram of dimension (N_MELS, N_SPECTRO) and subsequently into immage of dimension (N_MELS, N_SPECTRO) pixels'''
    mels = melspectrogram(signal, SR=params.sample_rate, N_MELS=params.mel_bands, N_FFT=params.n_fft, HOP_LENGTH=params.hop_length)
    if mono:
        # min-max scale to fit inside 8-bit range
        img = scale_minmax(mels, 0, 255).astype(np.uint16)
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        img = 255 - img  # invert. make black==more energy
        # save as PNG
        skimage.io.imsave(out, img)
    else:
        newImg = mono_to_color(mels)
        newImg = open_fat_image(newImg)
        newImg.show()
        newImg.save(out)
