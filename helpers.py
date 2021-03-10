""" REPOSITORY OF HELPERS FUNCTIONS """

import os
import numpy as np
import librosa
from tqdm import tqdm
import multiprocessing
import math
import warnings
from datetime import datetime
from itertools import islice


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



######################################################################
#
# HELPER FUNCTIONS
#
######################################################################


def dictChunked(it, size):
    ''' slice a dictionary in chunck to iterate'''
    it = iter(it)
    while True:
        p = tuple(islice(it, size))
        if not p:
            break
        yield p

def mergedict(*args):
    ''' merge more dictionaries'''
    output = {}
    for arg in args:
        output.update(arg)
    return output

def datetimeConverter(o):
    '''convert datetime into string format'''
    if isinstance(o, datetime):
        return o.__str__()


def cast_list(test_list, data_type):
    '''type cast list'''
    return list(map(data_type, test_list))


def cast_matrix(test_matrix, data_type):
    '''type cast matrix'''
    return list(map(lambda sub: list(map(data_type, sub)), test_matrix))


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


def yoloSetConverter(input_images, input_annotations):
    ''' convert COCO-notation in txt file valid for YOLO training'''
    for image in tqdm(input_images):
        image_id = image["id"]
        image_url = image["coco_url"]
        anno = image_url
        for annotation in input_annotations:
            if annotation["image_id"] == image_id:
                cat_id = annotation["category_id"]
                xmin = int(annotation["bbox"][0])
                xmax = int(annotation["bbox"][0] + annotation["bbox"][2])
                ymin = annotation["bbox"][1] - annotation["bbox"][3]
                ymax = annotation["bbox"][1]
                anno += ' ' + ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(cat_id)])
        f.write(anno + "\n")


def buildAudioDict(input_local_dir: list, input_cloud_dir: list, output_bucket_name: str, output_prefix: str):
    '''retrieve all the relevant information of an audio file as {fileName: (gcp_outputs_uri, local_outputs_uri, (annotations)}'''
    ''' usually works well combined with extract_from_bucket_v2 that output all relevant info abount local and cloud dir '''
    audioMappingDict = {}
    for filePath, blobPath in zip(input_local_dir, input_cloud_dir):
        listWords = []
        fileCompleteName, fileDir = os.path.split(filePath)[-1], os.path.split(filePath)[0]
        blobDir = os.path.split(blobPath)[0]
        if os.path.splitext(fileCompleteName)[-1] != ".txt":
            continue
        else:
            fileName = os.path.splitext(f"{fileCompleteName}")[0].rsplit('_', 1)[0]
            gcp_outputs_uri = f"gs://{output_bucket_name}/{output_prefix}/{fileName}"
            local_uri = f"{fileDir}/{fileName}"
            gcp_uri = f"{blobDir}/{fileName}"
            for line in open(f"{filePath}", "r"):
                listWords.append(line.rstrip("\n").split("\t"))
            audioMappingDict[(f"{fileName}")] = (
            f"{gcp_outputs_uri}", f"{local_uri}", f"{gcp_uri}", cast_matrix(listWords,float))
    return audioMappingDict