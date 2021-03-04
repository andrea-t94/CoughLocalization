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
credential_path = "/Users/andreatamburri/Documents/voicemed-d9a595992992.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

#input variables
input_bucket_name = 'voicemed-ml-raw-data'
output_bucket_name = 'voicemed-ml-processed-data'
prefix = "COUGHVIDannotated"
cocoSetName = "voicemedCocoSet"
annotation_master_dir = '/Users/andreatamburri/Desktop/tmp'


if __name__ == '__main__':

    # loading spectrogram extraction params
    params = spectro_params.Params()

    #GCP bucket prefixes
    cough_prefix = f"{prefix}"
    images_prefix = f"{prefix}/{params.mel_bands}_mels/images"
    annotation_prefix = f"{prefix}/{params.mel_bands}_mels/cocoset"
    dataset_prefix = f"{prefix}/{params.mel_bands}_mels/trainvalSet"

    #tmp paths
    image_path = fr'{annotation_master_dir}/{params.mel_bands}_mels/images'
    annotation_path = fr'{annotation_master_dir}/{params.mel_bands}_mels/coco_notations'
    dataset_path = fr'{annotation_master_dir}/{params.mel_bands}_mels/trainvalSet'
    tmp_dirs = [image_path, annotation_path, dataset_path]

    #cough extraction
    storage_client = storage.Client()
    input_bucket = storage_client.get_bucket(input_bucket_name)
    output_bucket = storage_client.get_bucket(output_bucket_name)
    extracted, blob_names = extract_from_bucket_v2(input_bucket.name, cough_prefix, root_path=annotation_master_dir)

    #tmp dirs creation
    for dir in tmp_dirs:
        try:
            os.makedirs(f"{dir}")
        except:
            shutil.rmtree(f"{dir}")
            os.makedirs(f"{dir}")

    ######
    # AudioDict struct
    # fileName : (gcp_artifacts_uri, local_path, xmin, xmax)
    ######
    audioDict = {}
    for filePath, blobPath in zip(extracted,blob_names):
        listWords = []
        file, fileDir = os.path.split(filePath)[-1], os.path.split(filePath)[0]
        blobDir = os.path.split(blobPath)[0]
        if os.path.splitext(file)[-1] != ".txt":
            continue
        else:
            fileName = os.path.splitext(f"{file}")[0].rsplit('_', 1)[0]
            gcp_artifacts_uri = f"gs://{blobDir}/{fileName}"
            local_artifacts_uri = f"{fileDir}/{fileName}"
            for line in open(f"{filePath}", "r"):
                listWords.append(line.rstrip("\n").split("\t"))
            audioDict[(f"{fileName}")] = (
            f"{gcp_artifacts_uri}", f"{local_artifacts_uri}", cast_matrix(listWords,float))

    #image and cocoSet processing
    images = []
    annotations = []
    # init ray
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    ray.put(audioDict)
    actors = [Annotator.remote(params, i) for i in range(num_cpus)]
    test_time = datetime.now()
    print('Processing spetro images and building annotations')
    for chunk in chunked(audioDict.items(), size=num_cpus):
        total = ray.get([actor.annotation_factory.remote(fileName,fileInfo,image_path)
                 for actor, (fileName,fileInfo) in zip(actors,chunk)])
        images_tmp, annotations_tmp = zip(*total)
        images_tmp = list(itertools.chain(*list(images_tmp)))
        annotations_tmp = list(itertools.chain(*list(annotations_tmp)))
    for image_tmp, annotation_tmp in zip(images_tmp,annotations_tmp):
        images.append(image_tmp)
        annotations.append(annotation_tmp)
    print(images)
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
        for image in tqdm(images):
            print(image)
            image_id = image["id"]
            image_url = image["coco_url"]
            anno = image_url
            for annotation in annotations:
                if annotation["image_id"] == image_id:
                    cat_id = annotation["category_id"]
                    xmin = int(annotation["bbox"][0])
                    xmax = int(annotation["bbox"][0] + annotation["bbox"][2])
                    ymin = annotation["bbox"][1] - annotation["bbox"][3]
                    ymax = annotation["bbox"][1]
                    anno += ' ' + ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(cat_id)])
            f.write(anno + "\n")

    #upload to GCP buckets
    upload_to_bucket_v2(output_bucket_name, images_prefix, root_path= image_path)
    upload_to_bucket_v2(output_bucket_name, annotation_prefix, root_path=annotation_path)
    upload_to_bucket_v2(output_bucket_name, dataset_prefix, root_path=dataset_path)
