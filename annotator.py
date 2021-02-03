''' Annotator for automatic creation of spectrogram images of dimension (N_MELS, N_SPECTRO)
    and creation of a COCO format dataset for cough localization '''

from helpers import SR, N_FFT, HOP_LENGTH, N_MELS
from helpers import myconverter, spectrogram_image

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
import uuid
import json

if __name__ == '__main__':

    annotation_master_dir = r'C:/Users/Administrator/Desktop/Voicemed/Annotation_Project/AVC_dump_annotated'
    file_path = r'C:/Users/Administrator/Desktop/Voicemed/Annotation_Project/preview_headset'
    image_path = r'C:/Users/Administrator/Desktop/Voicemed/Annotation_Project/cough_images'
    notation_path = r'C:/Users/Administrator/Desktop/Voicemed/Annotation_Project/coco_notations'
    #creating images path
    try:
        os.mkdir(f"{annotation_master_dir}/images")
    except:
        pass

    # creation COCO-wise annotation
    info = {
        "description": "VOICEMED Cough Dataset",
        "url": "",
        "version": "1.0.0",
        "year": 2020,
        "contributor": "Voicemed ML Team",
        "date_created": date.today()
    }

    licenses = [
        {
            "url": "https://www.voicemed.io/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ]

    categories = [
        {"supercategory": "human_sound", "id": 1, "name": "cough"},
        {"supercategory": "human_sound", "id": 2, "name": "breath"},
        {"supercategory": "human_sound", "id": 3, "name": "speech"},
        {"supercategory": "other", "id": 4, "name": "other"}
    ]

    images = [
    ]

    annotations = [
    ]

    all_files = os.listdir(annotation_master_dir)
    audioDict = {}
    for file in all_files:
        if  os.path.splitext(file)[-1] != ".txt":
            continue
        else:
            for line in open(f"{annotation_master_dir}/{file}", "r"):
                listWords = line.rstrip("\n").split("\t")
            audioDict[os.path.splitext(f"{file}")[0]] = (f"{annotation_master_dir}/{file}", float(listWords[0]),float(listWords[-1]))

    for key, value in audioDict.items():
        audio = os.path.splitext(value[0])[0] + ".wav"
        start_event = value[1]  # milliseconds
        end_event = value[-1]
        signal, rate = librosa.load(audio, sr=SR)

        # convert to PNG
        fileName = f"{key}.png"
        out = fr"{annotation_master_dir}/images/{fileName}"
        spectrogram_image(signal, sr=SR, out=out, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mels=N_MELS)

        N_SPECTRO = len(signal) / HOP_LENGTH
        # ration spectro:pixel is 1:1
        starting_point = start_event * SR / HOP_LENGTH #events in seconds
        ending_point = end_event * SR / HOP_LENGTH

        imageUuid = uuid.uuid4().hex
        image = {
                "license": 1,
                "file_name": fileName,
                "coco_url": out,
                "height": N_MELS,
                "width": N_SPECTRO,
                "date_captured": datetime.now(tz=None),
                "id": imageUuid
        }

        annotation = {
                "iscrowd": 0,  # just one image
                "image_id": imageUuid,  # same id as before
                "bbox": [starting_point, N_MELS, ending_point - starting_point, N_MELS],
                # top left x & y position, width and height
                "category_id": 1,  # stating for cough
                "id": uuid.uuid1()
        }

        images.append(image)
        annotations.append(annotation)

    #build-up the COCO-dataset
    voicemedCocoSet = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(fr'{annotation_master_dir}/voicemedCocoSet.json', 'w') as json_file:
        json_dump = json.dump(voicemedCocoSet, json_file, default=myconverter)
