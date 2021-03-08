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
from tf_features_extractor import Annotator
from gcp_utils import extract_from_bucket_v2, upload_to_bucket_v2
import uuid



def chunked(it, size):
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
input_bucket_name = 'voicemed-ml-raw-data'
output_bucket_name = 'voicemed-ml-processed-data'
prefix = "COUGHVIDannotated"
cocoSetName = "voicemedCocoSet"
annotation_master_dir = '/Users/andreatamburri/Desktop/test'
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

    #tmp dirs creation
    for dir in tmp_dirs:
        try:
            os.makedirs(f"{dir}")
        except:
            shutil.rmtree(f"{dir}")
            os.makedirs(f"{dir}")


    with open(f'{annotation_master_dir}/audioDict.txt', 'r') as file:
        audioDict = json.load(file)
    print(audioDict['3116448f-f316-4939-a492-0518f3dd06d7'])

    # image and cocoSet processing
    images = []
    annotations = []
    # init ray
    num_cpus = psutil.cpu_count(logical=False)
    print(num_cpus)
    ray.init(num_cpus=num_cpus)
    ray.put(audioDict)
    actors = [Annotator.remote(params, i) for i in range(num_cpus)]

    test_time = datetime.now()
    print('Processing spetro images and building annotations')
    for chunk in chunked(audioDict.items(), size=num_cpus):
        result = ray.get([actor.annotation_factory.remote(fileName, fileInfo, image_path)
                         for actor, (fileName, fileInfo) in zip(actors, chunk)])
        #unpack annotator results
        images_tmp, annotations_tmp = zip(*result)
        images_tmp = list(chain(*list(images_tmp)))
        annotations_tmp = list(chain(*list(annotations_tmp)))
        for image_tmp in images_tmp:
            images.append(image_tmp)
        for annotation_tmp in annotations_tmp:
            annotations.append(annotation_tmp)
        print(len(images))
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

    # building training-validation set as txt file of path,xmin,xmax,ymin,ymax
    with open(f"{dataset_path}/{cocoSetName}.txt", 'w') as f:
        yoloSetConverter(input_images=images, input_annotations=annotations)

    # upload toGCP buckets
    upload_to_bucket_v2(output_bucket_name, images_prefix, root_path=image_path)
    upload_to_bucket_v2(output_bucket_name, annotation_prefix, root_path=annotation_path)
    upload_to_bucket_v2(output_bucket_name, dataset_prefix, root_path=dataset_path)
