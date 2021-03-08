import os
import ray
import shutil
import psutil
import json
import itertools
from datetime import datetime
from tqdm import tqdm
from google.cloud import storage
import params as spectro_params
from cocoSet_params import info, licenses, categories
from helpers import datetimeConverter, cast_matrix
from tf_features_extractor import Annotator
from gcp_utils import extract_from_bucket_v2, upload_to_bucket_v2



def chunked(it, size):
    it = iter(it)
    while True:
        p = tuple(itertools.islice(it, size))
        if not p:
            break
        yield p

#credentials
credential_path = r"C:\Users\Administrator\Documents/voicemed-d9a595992992.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

#input variables
input_bucket_name = 'voicemed-ml-raw-data'
output_bucket_name = 'voicemed-ml-processed-data'
prefix = "COUGHVIDannotated"
cocoSetName = "voicemedCocoSet"
annotation_master_dir = r'C:\Users\Administrator\Desktop\tmp'
version_number = 3



if __name__ == '__main__':

    # loading spectrogram extraction params
    params = spectro_params.Params()

    #GCP bucket prefixes
    cough_prefix = f"{prefix}"
    images_prefix = f"{prefix}/v_{version_number}/{params.mel_bands}_mels/images"
    annotation_prefix = f"{prefix}/v_{version_number}/{params.mel_bands}_mels/cocoset"
    dataset_prefix = f"{prefix}/v_{version_number}/{params.mel_bands}_mels/trainvalSet"

    #tmp paths
    image_path = fr'{annotation_master_dir}/{params.mel_bands}_mels/images'
    annotation_path = fr'{annotation_master_dir}/{params.mel_bands}_mels/coco_notations'
    dataset_path = fr'{annotation_master_dir}/{params.mel_bands}_mels/trainvalSet'
    tmp_dirs = [image_path, annotation_path, dataset_path]

    #cough extraction
    storage_client = storage.Client()
    input_bucket = storage_client.get_bucket(input_bucket_name)
    output_bucket = storage_client.get_bucket(output_bucket_name)

    ######
    # AudioDict struct
    # fileName : (gcp_artifacts_uri, local_path, xmin, xmax)
    ######
    audioDict = {}
    for filePath in os.listdir(f"{annotation_master_dir}/{prefix}"):
        listWords = []
        file, fileDir = os.path.split(filePath)[-1], os.path.split(filePath)[0]
        #blobDir = os.path.split(blobPath)[0]
        if os.path.splitext(file)[-1] != ".txt":
            continue
        else:
            fileName = os.path.splitext(f"{file}")[0].rsplit('_', 1)[0]
            gcp_artifacts_uri = f"gs://test/{fileName}"
            local_artifacts_uri = f"{fileDir}/{fileName}"
            for line in open(f"{filePath}", "r"):
                listWords.append(line.rstrip("\n").split("\t"))
            audioDict[(f"{fileName}")] = (
            f"{gcp_artifacts_uri}", f"{local_artifacts_uri}", cast_matrix(listWords,float))

    with open(f'{annotation_master_dir}/audioDict.txt', 'w') as file:
        file.write(json.dumps(audioDict))
    print(audioDict['3116448f-f316-4939-a492-0518f3dd06d7'])