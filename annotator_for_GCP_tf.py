import os
import ray
import shutil
import psutil
import json
from google.cloud import storage
from itertools import chain, zip_longest
from datetime import datetime

import params as spectro_params
from cocoSet_params import info, licenses, categories
from helpers import datetimeConverter, dictChunked, buildAudioDict
from tf_features_extractor import Annotator
from gcp_utils import extract_from_bucket_v2, upload_to_bucket_v2


#credentials
credential_path = "/Users/andreatamburri/Documents/voicemed-d9a595992992.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

#input variables
input_bucket_name = 'voicemed-ml-raw-data'
output_bucket_name = 'voicemed-ml-processed-data'
prefix = "COUGHVIDannotated"
cocoSetName = "voicemedCocoSet"
annotation_master_dir = '/Users/andreatamburri/Desktop/tmp'
version_number = 3
features = ['spectrogram']



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
    local_dirs, gcp_dirs = extract_from_bucket_v2(input_bucket.name, prefix, root_path=annotation_master_dir)

    #tmp dirs creation
    for dir in tmp_dirs:
        try:
            os.makedirs(f"{dir}")
        except:
            shutil.rmtree(f"{dir}")
            os.makedirs(f"{dir}")

    ######
    # AudioDict struct
    # fileName : (gcp_output_uri, local_path, gcp_path, (xmin, xmax))
    ######
    audioDict = buildAudioDict(local_dirs, gcp_dirs, output_bucket_name images_prefix)

    #image and cocoSet processing
    images = []
    annotations = []
    # init ray
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    ray.put(audioDict)
    actors = [Annotator.remote(params, feature, i) for i in range(num_cpus)]
    test_time = datetime.now()
    print('Processing spetro images and building annotations')
    for chunk in dictChunked(audioDict.items(), size=num_cpus):
        total = ray.get([actor.annotation_factory.remote(fileName,fileInfo,image_path)
                 for actor, (fileName,fileInfo) in zip(actors,chunk)])
        images_tmp, annotations_tmp = zip(*total)
        images_tmp = list(chain(*list(images_tmp)))
        annotations_tmp = list(chain(*list(annotations_tmp)))
        for image_tmp in images_tmp:
            images.append(image_tmp)
        for annotation_tmp in annotation_tmp:
            annotations.append(annotation_tmp)
    print(f"Total time is {datetime.now() - test_time}")

    # build-up the COCO-dataset
    voicemedCocoSet = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(fr'{annotation_path}/{cocoSetName}.json', 'w') as json_file:
        json_dump = json.dump(voicemedCocoSet, json_file, default=datetimeConverter)

    #building training-validation set as txt file of path,xmin,xmax,ymin,ymax
    with open(f"{dataset_path}/{cocoSetName}.txt", 'w') as f:
        yoloSetConverter(input_images=images, input_annotations=annotations)

    #upload to GCP buckets
    upload_to_bucket_v2(output_bucket_name, images_prefix, root_path= image_path)
    upload_to_bucket_v2(output_bucket_name, annotation_prefix, root_path=annotation_path)
    upload_to_bucket_v2(output_bucket_name, dataset_prefix, root_path=dataset_path)
