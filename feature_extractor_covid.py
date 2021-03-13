# from audio data enriched with annotation the script extracts and stores:
# - cropped audio samples
# - features, as spectrograms and mfccs, in numpy arrays
# - training-validtion set

import os
import ray
import shutil
import psutil
from datetime import datetime
from google.cloud import storage
import json
from itertools import chain
import numpy as np

import params as spectro_params
from tf_features_extractor import Extractor, Cropper
from gcp_utils import extract_from_bucket_v2, upload_to_bucket_v2
from helpers import buildAudioDict, dictChunked

# credentials
credential_path = "/Users/andreatamburri/Documents/voicemed-d9a595992992.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# input variables
version_number = 1
input_bucket_name = 'voicemed-ml-raw-data'
output_bucket_name = 'voicemed-ml-processed-data'
prefix = "COUGHVIDannotated" #prefix of the data source to extract
covidSetName = "voicemedCovidSet"
features_to_compute = ['mfcc']

if __name__ == '__main__':

    coughvidLabelMap = {
        'healthy': 0,
        'COVID19': 1
    }

    # loading spectrogram extraction params
    params = spectro_params.Params()

    # loading Extractor
    features_extractor = Extractor(params, features_to_compute, 1)

    # GCP bucket prefixes
    cough_prefix = f"{prefix}/covidClass/{version_number}"
    crop_prefix = f"{cough_prefix}/cropped_samples"
    spectro_prefix = f"{cough_prefix}/spectrograms/{params.mel_bands}_mels/raw_spectro"
    mfcc_prefix = f"{cough_prefix}/mfccs/{params.n_mfcc}_n_mfcc/raw_mfcc"
    spectro_dataset_prefix = f"{cough_prefix}/spectrograms/{params.mel_bands}_mels/trainvalSet"
    mfcc_dataset_prefix = f"{cough_prefix}/mfccs/{params.n_mfcc}_n_mfcc/trainvalSet"

    # tmp paths
    annotation_master_dir = fr'/Users/andreatamburri/Desktop/tmp/covidClass/v_{version_number}'
    crop_path = fr'{annotation_master_dir}/crop_samples'
    spectro_path = fr'{annotation_master_dir}/spectrograms/{params.mel_bands}_mels/raw_spectro'
    mfcc_path = fr'{annotation_master_dir}/mfccs/{params.n_mfcc}_n_mfcc/raw_mfcc'
    spectro_dataset_path = fr'{annotation_master_dir}/spectrograms/{params.mel_bands}_mels/trainvalSet'
    mfcc_dataset_path = fr'{annotation_master_dir}/mfccs/{params.n_mfcc}_n_mfcc/trainvalSet'
    tmp_dirs = [crop_path, spectro_path, mfcc_path, spectro_dataset_path, mfcc_dataset_path]

    # cough extraction
    # storage_client = storage.Client()
    # input_bucket = storage_client.get_bucket(input_bucket_name)
    # output_bucket = storage_client.get_bucket(output_bucket_name)
    # local_dirs, gcp_dirs = extract_from_bucket_v2(input_bucket.name, prefix, root_path=annotation_master_dir)
    #
    # # tmp dirs creation
    # for dir in tmp_dirs:
    #     try:
    #         os.makedirs(f"{dir}")
    #     except:
    #         shutil.rmtree(f"{dir}")
    #         os.makedirs(f"{dir}")

    ######
    # AudioDict struct
    # fileName : (gcp_output_uri, local_path, gcp_path, annotation (xmin, xmax))
    ######
    # audioDict = buildAudioDict(local_dirs, gcp_dirs, output_bucket_name, crop_prefix)
    # with open(fr'{annotation_master_dir}/audioDict.txt', 'w') as file:
    #     file.write(json.dumps(audioDict))

    with open(fr'{annotation_master_dir}/audioDict.txt', 'r') as file:
        audioDict = json.load(file)
    # image cropping and feature extraction: Spectrograms and Mfccs
    trainValSetMfcc = []
    trainValSetSpectro = []
    # init ray
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    ray.put(coughvidLabelMap)
    actors = [Cropper.remote(params, features_to_compute, i) for i in range(num_cpus)]
    test_time = datetime.now()
    print('cropping and feature extraction of audio files')
    for chunk in dictChunked(audioDict.items(), size=num_cpus):
        total = ray.get([actor.cropping_factory.remote(fileName, fileInfo, crop_path, coughvidLabelMap,
                                                       mfcc_path, spectro_path)
                         for actor, (fileName, fileInfo) in zip(actors, chunk)])
        trainValSetMfccTmp, trainValSetSpectroTmp = zip(*total)
        trainValSetMfccTmp = list(chain(*list(trainValSetMfccTmp)))
        trainValSetSpectroTmp = list(chain(*list(trainValSetSpectroTmp)))
        for lineMfcc in trainValSetMfccTmp:
            trainValSetMfcc.append(lineMfcc)
        for lineSpectro in trainValSetSpectroTmp:
            trainValSetSpectro.append(lineSpectro)
    #save mfcc
    trainValSetMfcc = np.array(trainValSetMfcc)
    np.random.shuffle(trainValSetMfcc)
    np.save(f"{mfcc_dataset_path}/trainValSet.npy", trainValSetMfcc)
    #sace spectrogram
    trainValSetSpectro = np.array(trainValSetSpectro)
    np.random.shuffle(trainValSetSpectro)
    np.save(f"{spectro_dataset_path}/trainValSet.npy", trainValSetMfcc)

    #upload toGCP buckets
    upload_to_bucket_v2(output_bucket_name, crop_prefix, root_path= crop_path)
    upload_to_bucket_v2(output_bucket_name, spectro_prefix, root_path=spectro_path)
    upload_to_bucket_v2(output_bucket_name, spectro_dataset_prefix, root_path=spectro_dataset_path)
    upload_to_bucket_v2(output_bucket_name, mfcc_prefix, root_path=mfcc_path)
    upload_to_bucket_v2(output_bucket_name, mfcc_dataset_prefix, root_path=mfcc_dataset_path)
