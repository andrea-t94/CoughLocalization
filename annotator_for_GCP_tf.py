import os
import shutil
import multiprocessing
import warnings
import json
import uuid
from datetime import datetime
from tqdm import tqdm
from google.cloud import storage
import params as spectro_params
from cocoSet_params import info, licenses, categories
from helpers import datetimeConverter, cast_matrix
from audio_utils import spectrogram_image_tf
from gcp_utils import extract_from_bucket_v2, upload_to_bucket_v2


def annotation_factory(
    fileName: str,
    fileInfo: tuple,
    image_path: str,
    params: object,
    annotations: list,
    images: list
):
    audio = fileInfo[1] + ".wav"
    annotation_events = fileInfo[-1]

    # convert to PNG
    fileNameOut = f"{fileName}.png"
    out = fr"{image_path}/{fileNameOut}"
    len_signal, spectro_shape = spectrogram_image_tf(audio, params=params, out=out)

    N_MELS, N_SPECTRO = spectro_shape[0], spectro_shape[1]
    frames_per_spectro = len_signal / N_SPECTRO
    # spectro:pixel is 1:1
    # in this way I find how many frames are contained in a pixel
    # in order to say how many pixels are in the bounding boxes
    imageUuid = uuid.uuid4().hex
    image = {
        "license": 1,
        "file_name": fileName,
        "coco_url": f"{fileInfo[0]}.png",
        "height": N_MELS,
        "width": N_SPECTRO,
        "date_captured": datetime.now(tz=None),
        "id": imageUuid
    }
    images.append(image)

    for event in annotation_events:
        starting_event, ending_event = event[0], event[-1]
        starting_point = starting_event * params.sample_rate / frames_per_spectro #events in seconds
        ending_point = ending_event * params.sample_rate / frames_per_spectro
        annotation = {
            "iscrowd": 0,
            "image_id": imageUuid,
            "bbox": [starting_point, N_MELS, ending_point - starting_point, N_MELS],
            # top left x & y position, width and height
            "category_id": 0,
            "id": uuid.uuid1()
        }
        annotations.append(annotation)

    return images, annotations

#credentials
credential_path = "C:/Users/Administrator/Documents/voicemed-d9a595992992.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

#input variables
input_bucket_name = 'voicemed-ml-raw-data'
output_bucket_name = 'voicemed-ml-processed-data'
prefix = "COUGHVIDannotated"
cocoSetName = "voicemedCocoSet"
annotation_master_dir = r'C:/Users/Administrator/Desktop/tmp'


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

    manager = multiprocessing.Manager()
    threads = []
    images = manager.list()
    annotations = manager.list()
    #image and cocoSet processing
    test_time = datetime.now()
    for key, value in audioDict.items():
        p = multiprocessing.Process(target=annotation_factory,
                                    args=(key, value, image_path, params, annotations, images))
        threads.append(p)
    print('Processing spetro images and building annotations')
    with tqdm(total=len(threads)) as pbar:
        cores = multiprocessing.cpu_count()
        n = cores
        while len(threads) > 0:
            if len(threads) < n:
                n = len(threads)
                warnings.warn(
                    f"Low amount of files to process, lower than number of CPU cores, consisting of {n}",
                    ResourceWarning)
            for i in range(n):
                try:
                    threads[i].start()
                except:
                    warnings.warn(f"Low amount of files to process, lower than number of CPU cores, consisting of {n}",
                                  ResourceWarning)
                    n = len(threads)
                    pass
            for i in range(n):
                threads[i].join()
                pbar.update(1)

            threads = threads[n:]

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

    #upload toGCP buckets
    upload_to_bucket_v2(output_bucket_name, images_prefix, root_path= image_path)
    upload_to_bucket_v2(output_bucket_name, annotation_prefix, root_path=annotation_path)
    upload_to_bucket_v2(output_bucket_name, dataset_prefix, root_path=dataset_path)
