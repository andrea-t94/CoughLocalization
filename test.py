import os
import ray
import shutil
import psutil
import json
from itertools import chain, islice
from datetime import datetime
from tqdm import tqdm
from google.cloud import storage
import params as spectro_params
from cocoSet_params import info, licenses, categories
from helpers import datetimeConverter, cast_matrix
from tf_features_extractor import Extractor, Cropper
from gcp_utils import extract_from_bucket_v2, upload_to_bucket_v2
import uuid
import soundfile as sf
import numpy as np
from helpers import buildAudioDict, dictChunked


def dictChunked(it, size):
    it = iter(it)
    while True:
        p = tuple(islice(it, size))
        if not p:
            break
        yield p


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


#credentials
credential_path = "/Users/andreatamburri/Documents/voicemed-d9a595992992.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    

#input variables
version_number = 1
input_bucket_name = 'voicemed-ml-raw-data'
output_bucket_name = 'voicemed-ml-processed-data'
prefix = "kevin"
covidSetName = "voicemedCovidSet"
annotation_master_dir = fr'/Users/andreatamburri/Desktop/kevin/covidClass/v_{version_number}'
features_to_compute = ['mfcc']

if __name__ == '__main__':

    coughvidLabelMap = {
        'healthy': 0,
        'COVID19': 1
    }

    #loading spectrogram extraction params
    params = spectro_params.Params()

    #loading Extractor
    features_extractor = Extractor(params, features_to_compute, 1)

    #GCP bucket prefixes
    cough_prefix = f"{prefix}/covid/{version_number}"
    crop_prefix = f"{prefix}/cropped_samples"
    spectro_prefix = f"{prefix}/spectrograms/{params.mel_bands}_mels/raw_spectro"
    mfcc_prefix = f"{prefix}/mfccs/{params.n_mfcc}_n_mfcc/raw_mfcc"
    spectro_dataset_prefix = f"{prefix}/spectrograms/{params.mel_bands}_mels/trainvalSet"
    mfcc_dataset_prefix = f"{prefix}/mfccs/{params.n_mfcc}_n_mfcc/trainvalSet"

    #tmp paths
    crop_path = fr'{annotation_master_dir}/crop_samples'
    spectro_path = fr'{annotation_master_dir}/spectrograms/{params.mel_bands}_mels/raw_spectro'
    mfcc_path = fr'{annotation_master_dir}/mfccs/{params.n_mfcc}_n_mfcc/raw_mfcc'
    spectro_dataset_path = fr'{annotation_master_dir}/spectrograms/{params.mel_bands}_mels/trainvalSet'
    mfcc_dataset_path = fr'{annotation_master_dir}/mfccs/{params.n_mfcc}_n_mfcc/trainvalSet'
    tmp_dirs = [crop_path, spectro_path, mfcc_path, spectro_dataset_path, mfcc_dataset_path]


    # cough extraction
    storage_client = storage.Client()
    input_bucket = storage_client.get_bucket(input_bucket_name)
    output_bucket = storage_client.get_bucket(output_bucket_name)
    local_dirs, gcp_dirs = extract_from_bucket_v2(input_bucket.name, prefix, root_path=annotation_master_dir)

    #tmp dirs creation
    for dir in tmp_dirs:
        print(dir)
        os.makedirs(f"{dir}")

    audioDict = buildAudioDict(local_dirs, gcp_dirs, output_bucket_name, crop_prefix)
    with open(fr'{annotation_master_dir}/audioDict.txt', 'w') as file:
         file.write(json.dumps(audioDict))
    with open(fr'{annotation_master_dir}/audioDict.txt', 'r') as file:
        audioDict = json.load(file)

    ######
    # CropAudioDict struct
    # cropFilePath :(fileName, labelName, label)
    ######

    # for key,val in audioDict.items():
    #     #info retrieval
    #     filePath = f'{val[1]}.wav'
    #     gcpPath = val[0]
    #     fileName = key
    #     annotations = val[-1]
    #     labelName = gcpPath.split(f"{prefix}")[-1].split("/")[1]
    #     label = coughvidLabelMap[labelName]
    #     # Decode the WAV file.
    #     signal, sr = sf.read(filePath, dtype=np.int16)
    #     for i, annotation in enumerate(annotations):
    #         starting_point = int(annotation[0]*sr)
    #         ending_point = int(annotation[1]*sr)+1
    #         crop_audio_path = f'{crop_path}/{labelName}/{fileName}_{i}.wav'
    #         signal1, sr = sf.read(filePath, dtype=np.int16, start=starting_point, stop=ending_point)
    #         try:
    #             sf.write(crop_audio_path, signal1, sr)
    #         except:
    #             os.makedirs(f"{crop_path}/{labelName}")
    #             sf.write(crop_audio_path, signal1, sr)
    #         cropAudioDict[crop_audio_path] = (f'{fileName}_{i}',labelName, label)

    # image cropping
    trainValSetMfcc = []
    trainValSetSpectro = []
    # init ray
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    ray.put(coughvidLabelMap)
    actors = [Cropper.remote(params, features_to_compute, i) for i in range(num_cpus)]
    test_time = datetime.now()
    print('cropping audio files')
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

    print(len(trainValSetMfcc))
    trainValSetMfcc = np.array(trainValSetMfcc)
    np.random.shuffle(trainValSetMfcc)
    np.save(f"{mfcc_dataset_path}/trainValSet.npy",trainValSetMfcc)
    print(len(trainValSetSpectro))
    trainValSetSpectro = np.array(trainValSetSpectro)
    np.random.shuffle(trainValSetSpectro)
    np.save(f"{spectro_dataset_path}/trainValSet.npy", trainValSetSpectro)


    # trainValSet = []
    # for crop_audio, (cropFileName, labelName, label) in cropAudioDict.items():
    #     spectrogram, mfcc, features = features_extractor.feature_tf(crop_audio)
    #     np_mfcc = np.array([mfcc,label])
    #     try:
    #         np.save(f"{mfcc_path}/{labelName}/{cropFileName}.npy",np_mfcc)
    #     except:
    #         os.makedirs(f"{mfcc_path}/{labelName}")
    #         np.save(f"{mfcc_path}/{labelName}/{cropFileName}.npy",np_mfcc)
    #     trainValSet.append(np_mfcc)
    #
    # trainValSet = np.array(trainValSet)
    # np.random.shuffle(trainValSet)
    # np.save(f"{mfcc_dataset_path}/trainValSet.npy",trainValSet)


    #upload toGCP buckets
    upload_to_bucket_v2(output_bucket_name, crop_prefix, root_path= crop_path)
    upload_to_bucket_v2(output_bucket_name, spectro_prefix, root_path=spectro_path)
    upload_to_bucket_v2(output_bucket_name, spectro_dataset_prefix, root_path=spectro_dataset_path)
    upload_to_bucket_v2(output_bucket_name, mfcc_prefix, root_path=mfcc_path)
    upload_to_bucket_v2(output_bucket_name, mfcc_dataset_prefix, root_path=mfcc_dataset_path)
