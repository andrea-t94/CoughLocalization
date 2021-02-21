import os
import warnings
import librosa.display
from google.cloud import storage
import json
import uuid
from datetime import date, datetime
from helpers import spectrogram_image
from utils import extract_from_bucket_v2, upload_to_bucket_v2
import params as spectro_params

######################################################################
#
# FEATURE EXTRACTION FUNCTIONS
#
######################################################################

def myconverter(o):
    '''convert datetime into string format'''
    if isinstance(o, datetime):
        return o.__str__()

# helper function to type cast list
def cast_list(test_list, data_type):
    return list(map(data_type, test_list))

# helper function to type cast Matrix
def cast_matrix(test_matrix, data_type):
    return list(map(lambda sub: list(map(data_type, sub)), test_matrix))

if __name__ == '__main__':
    #credentials
    credential_path = "C:/Users/Administrator/Documents/voicemed-d9a595992992.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    bucket_name = 'voicemed-ml-sandbox'
    prefix = "Demo"
    cough_prefix = f"{prefix}/Demo"
    images_prefix = f"{prefix}/images"
    annotation_prefix = f"{prefix}/cocoset"
    dataset_prefix = f"{prefix}/trainvalSet"

    #tmp paths
    annotation_master_dir = r'C:/Users/Administrator/Desktop/tmp'
    image_path = fr'{annotation_master_dir}/images'
    annotation_path = fr'{annotation_master_dir}/coco_notations'
    dataset_path = fr'{annotation_master_dir}/trainvalSet'
    tmp_dirs = [annotation_master_dir, image_path, annotation_path, dataset_path]
    cocoSetName = "voicemedCocoSet"

    for dir in tmp_dirs:
    try:
        os.mkdir(f"{dir}")
    except:
        pass

    params = spectro_params.Params()

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

    #coguh extraction
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    extracted,blob_names = extract_from_bucket_v2(bucket.name,cough_prefix,root_path=annotation_master_dir,local_dir='')

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
    print(audioDict)

    #image and cocoSet processing
    for key, value in audioDict.items():
        audio = value[1] + ".wav"
        annotation_events = value[-1]
        signal, rate = librosa.load(audio, sr=params.sample_rate)

        # convert to PNG
        fileNameOut = f"{key}.png"
        out = fr"{annotation_master_dir}/images/{fileNameOut}"
        spectrogram_image(signal, params=params, out=out)

        N_SPECTRO = len(signal) / params.hop_length
        # ration spectro:pixel is 1:1
        imageUuid = uuid.uuid4().hex
        image = {
            "license": 1,
            "file_name": fileName,
            "coco_url": value[0],
            "height": params.mel_bands,
            "width": N_SPECTRO,
            "date_captured": datetime.now(tz=None),
            "id": imageUuid
        }
        images.append(image)

        for event in annotation_events:
            starting_event, ending_event = event[0], event[-1]
            starting_point = starting_event * params.sample_rate / params.hop_length #events in seconds
            ending_point = ending_event * params.sample_rate / params.hop_length
            annotation = {
                "iscrowd": 0,  # just one image
                "image_id": imageUuid,  # same id as before
                "bbox": [starting_point, params.mel_bands, ending_point - starting_point, params.mel_bands],
                # top left x & y position, width and height
                "category_id": 1,  # stating for cough
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
        json_dump = json.dump(voicemedCocoSet, json_file, default=myconverter)

    #building training-validation set as txt file of path,xmin,xmax,ymin,ymax,class
    with open(f"{dataset_path}/{cocoSetName}.txt", 'w') as f:
        for image in images:
            image_id = image["id"]
            image_path = image["coco_url"]
            anno = image_path
            for annotation in annotations:
                if annotation["image_id"] == image_id:
                    cat_id = annotation["category_id"]
                    xmin = round(annotation["bbox"][0], 2)
                    xmax = round(annotation["bbox"][0] + annotation["bbox"][2], 2)
                    ymin = annotation["bbox"][1] - annotation["bbox"][3]
                    ymax = annotation["bbox"][1]
                    anno += ' ' + ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(cat_id)])
            print(anno)
            f.write(anno + "\n")

    upload_to_bucket_v2(bucket_name, images_prefix, root_path= image_path, local_dir='')
    upload_to_bucket_v2(bucket_name, annotation_prefix, root_path=annotation_path, local_dir='', file=fr'{cocoSetName}.json')
    upload_to_bucket_v2(bucket_name, dataset_prefix, root_path=dataset_path, local_dir='', file=f'{cocoSetName}.txt')
