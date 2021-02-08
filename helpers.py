""" REPOSITORY OF HELPERS FUNCTIONS """
""" THE FILE CONTAINS FUNCTIONS FOR PREPROCESSING SOUND DATA FILES """
""" COMPRISING OF FEATURE EXTRACTIONS FUNCTIONS AS WELL AS FUNTIONS FOR COMMUNICATING WITH GCP ENVIRONMENT"""
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import librosa.display
import multiprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import warnings
import librosa.display
import skimage.io
from datetime import date, datetime
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img

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
LOCAL_DIR = "/home/jupyter/"
AUGMENT = LOCAL_DIR + "extracted-data/Noises/"


######################################################################
#
# FEATURE EXTRACTION FUNCTIONS
#
######################################################################

def myconverter(o):
    '''convert datetime into string format'''
    if isinstance(o, datetime):
        return o.__str__()


def envelope(signal, rate, thresh):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    # Create aggregated mean
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for m in y_mean:
        mask.append(m > thresh)

    return mask


def load_audio(path):
    signal, rate = librosa.load(path, sr=SR)
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


def melspectrogram(signal):
    signal = librosa.util.normalize(signal)
    spectro = librosa.feature.melspectrogram(
        signal,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length= HOP_LENGTH
    )
    spectro = librosa.power_to_db(spectro**2, ref=np.max)
    spectro = spectro.astype(np.float32)
    return spectro


def MFCC(sample):
    return librosa.feature.mfcc(sample, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH).T



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
        V = np.zeros_like(Xstd, dtype=np.uint16)
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

def spectrogram_image(signal, sr, out, hop_length, n_fft, n_mels, mono=True):
    ''' convert audio in logmel-spectrogram of dimension (N_MELS, N_SPECTRO) and subsequently into immage of dimension (N_MELS, N_SPECTRO) pixels'''
    mels = melspectrogram(signal)
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


def load_noises():
    noises = {}
    for cat in os.listdir(AUGMENT):
        if not os.path.isdir(AUGMENT + cat):
            continue
        noises[cat] = os.listdir(AUGMENT + cat + "/")
    ns = []
    for cat in noises:
        i = np.random.choice(len(noises[cat]))
        noise, _ = librosa.load(AUGMENT + cat + "/" + noises[cat][i], sr=SR)
        ns.append(noise)
    return ns



def process(audio, data, i, spectro, n_aug, pad=SAMPLE_SIZE):
    signal = load_audio(audio)

    if len(signal) < SAMPLE_SIZE:
        return []

    if n_aug > 0:
        ns = load_noises()

    start = 0
    end = SAMPLE_SIZE
    features = []

    while True:
        if end > len(signal):
            break
        else:
            sample = signal[start:end]
            start += PAD
            end += PAD

        if not spectro:
            mfcc = MFCC(sample)
            for frame in mfcc:  # this will be of length 14
                features.append(frame)
            if n_aug > 0:
                signals = augment(sample, ns, n_aug, spectro=spectro)
                for s in signals:
                    mfcc = MFCC(s)
                    for frame in mfcc:  # this will be of length 14
                        features.append(frame)

        else:

            features.append(melspectrogram(sample))

            if n_aug > 0:
                signals = augment(sample, ns, n_aug, spectro=spectro)
                for s in signals:
                    features.append(melspectrogram(s))

    for feat in features:
        data.append([feat, i])

    return data


def generate_dataset(folder, output_folder, output_name, spectro=True, start=0, batch=100000, n_aug=4,
                     save_to_bucket=True):
    """ Generate Spectrograms/Mfccs based on librosa library. The function need already labeled datasets exploiting multiprocessing management to speed up the performances."""

    manager = multiprocessing.Manager()
    threads = []
    n_audio = 0
    start_point = start

    try:
        os.mkdir(output_folder + "/")
    except:
        pass

    end_point = start_point + batch
    if not spectro:
        labels = ['cough', 'covid']
    else:
        labels = ['cough', 'not']

    for i, label in enumerate(labels):
        n_audio += len(os.listdir(folder + label))
    iterations = math.ceil(n_audio / (batch * len(labels)))
    print('preprocessing ' + folder + '...')

    for j in range(iterations):
        print('batch n: ' + str(j + 1))
        data = manager.list()  # contains [mel, label]
        for i, label in enumerate(labels):
            for audio in os.listdir(folder + label)[start_point:end_point]:
                if os.path.splitext(audio)[-1] != ".wav":
                    continue
                p = multiprocessing.Process(target=process,
                                            args=(folder + label + "/" + audio, data, i, spectro, n_aug))
                threads.append(p)

        cores = multiprocessing.cpu_count()
        n = cores
        with tqdm(total=len(threads)) as pbar:
            while len(threads) > 0:
                for i in range(n):
                    threads[i].start()

                for i in range(n):
                    threads[i].join()
                    pbar.update(1)

                threads = threads[n:]
                if len(threads) < n:
                    n = len(threads)

        data = np.array(list(data))
        np.random.shuffle(data)
        if not spectro:
            if n_aug > 0:
                output_filename = "{0}_mfcc_aug_{1}.npy".format(output_name, j + 1)
            else:
                output_filename = "{0}_mfcc_{1}.npy".format(output_name, j + 1)
        else:
            if n_aug > 0:
                output_filename = "{0}_spectrogram_aug_{1}.npy".format(output_name, j + 1)
            else:
                output_filename = "{0}_spectrogram_{1}.npy".format(output_name, j + 1)
        np.save(output_folder + "/" + output_filename, data)
        print(output_folder + "/" + output_filename)

        if save_to_bucket:
            bucket_name = 'voicemed-ml-processed-data'
            # output folder kept embedded local dir here in order to keep it more general
            root_path = os.path.split(output_folder)[1]
            upload_to_bucket(bucket_name=bucket_name, prefix=output_name, root_path=root_path, file=output_filename)

        start_point += batch
        end_point += batch
