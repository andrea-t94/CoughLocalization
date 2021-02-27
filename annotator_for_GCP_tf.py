import os
from google.cloud import storage
import json
import uuid
from datetime import datetime
from helpers import datetimeConverter, cast_matrix
from audio_utils import spectrogram_image_tf
from gcp_utils import extract_from_bucket_v2, upload_to_bucket_v2
import params as spectro_params
from tqdm import tqdm
from cocoSet_params import info, licenses, categories
import shutil

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
    #GCP bucket prefixes
    cough_prefix = f"{prefix}"
    images_prefix = f"{prefix}/images"
    annotation_prefix = f"{prefix}/cocoset"
    dataset_prefix = f"{prefix}/trainvalSet"

    #tmp paths
    image_path = fr'{annotation_master_dir}/images'
    annotation_path = fr'{annotation_master_dir}/coco_notations'
    dataset_path = fr'{annotation_master_dir}/trainvalSet'
    tmp_dirs = [annotation_master_dir, image_path, annotation_path, dataset_path]

    #tmp dirs creation
    for dir in tmp_dirs:
        try:
            os.mkdir(f"{dir}")
        except:
            shutil.rmtree(f"{dir}")
            os.mkdir(f"{dir}")

    #loading spectrogram extraction params
    params = spectro_params.Params()

    #cough extraction
    storage_client = storage.Client()
    input_bucket = storage_client.get_bucket(input_bucket_name)
    output_bucket = storage_client.get_bucket(output_bucket_name)
    extracted, blob_names = extract_from_bucket_v2(input_bucket.name, cough_prefix, root_path=annotation_master_dir,max_samples=50)
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

    images = []
    annotations = []
    #image and cocoSet processing
    for key, value in tqdm(audioDict.items()):
        audio = value[1] + ".wav"
        annotation_events = value[-1]

        # convert to PNG
        fileNameOut = f"{key}.png"
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
            "coco_url": value[0],
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
            print(anno)
            f.write(anno + "\n")

    #upload toGCP buckets
    upload_to_bucket_v2(output_bucket_name, images_prefix, root_path= image_path)
    upload_to_bucket_v2(output_bucket_name, annotation_prefix, root_path=annotation_path)
    upload_to_bucket_v2(output_bucket_name, dataset_prefix, root_path=dataset_path)
