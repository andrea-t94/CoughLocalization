""" REPOSITORY OF UTILS FUNCTIONS """
""" COMPRISING OF FEATURE EXTRACTIONS FUNCTIONS AS WELL AS FUNTIONS FOR COMMUNICATING WITH GCP ENVIRONMENT"""

import datetime
import os
import subprocess
import sys
import pandas as pd
import numpy as np
import shutil
from google.cloud import storage
from tqdm import tqdm
import multiprocessing
import math
import warnings
from google.cloud import storage

######################################################################
#
#ENVIRONMENTAL VARIABLES
#
######################################################################

warnings.filterwarnings("ignore")
SR = 44000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 60
SILENCE = 0.0018
SAMPLE_LENGTH = 0.5 #s
SAMPLE_SIZE = int(np.ceil(SR*SAMPLE_LENGTH))
NOISE_RATIO = 0.3
PAD = int(np.ceil(SR*0.05))
N_MFCC = 14
LOCAL_DIR = "/home/jupyter/"
AUGMENT = LOCAL_DIR+"extracted-data/Noises/"


######################################################################
#
#GCP INTERACTION FUNCTIONS
#
######################################################################

def upload_to_bucket(bucket_name, prefix, root_path='', file=None, local_dir="/home/jupyter/"):
    """ Upload file into a bucket, defined by bucket_name and prefix, the folder inside a bucket"""

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    upload_dir = local_dir+root_path

    if file!=None:
        """Uploads a file to the bucket."""
        blob = bucket.blob(prefix+"/"+file)
        blob.upload_from_filename(upload_dir+"/"+file)

        print('File {} uploaded to {}.'.format(
            file,
            bucket_name+"/"+prefix))

    else:
        """Uploads a directory to the bucket."""
        for root, dirs ,files in os.walk(upload_dir):
            for file in files:
                blob = bucket.blob(prefix+"/"+file)
                blob.upload_from_filename(root+"/"+file)

                print('File {} uploaded to {}.'.format(
                    file,
                    bucket_name+"/"+prefix))


def extract_from_bucket(bucket_name,prefix,root_path,local_dir="/home/jupyter/",file=None):
    """ Download file from a bucket, identified by bucket_name, prefix stands for folder inside the bucket"""

    try:
        os.mkdir(root_path+"/")
    except:
        pass
    extracted=[]
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    if file!=None:
        """ Download an entire folder """
        blob = bucket.blob(prefix+'/'+file)
        filename = blob.name.replace('_', '/')
        new_dir = os.path.split(filename)[0]
        new_path = root_path
        for path in new_dir.split('/'):
            new_path += '/'+path
            try:
                os.mkdir(new_path)
            except:
                pass
        blob.download_to_filename(local_dir + root_path + "/" + filename)
        print(local_dir + filename)
        extracted.append(local_dir + root_path + "/" + filename)

    else:
        """ Download a single file """
        blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
        for blob in blobs:
            filename = blob.name.replace('_', '/')
            new_dir = os.path.split(filename)[0]
            new_path = root_path
            for path in new_dir.split('/'):
                new_path += '/'+path
                try:
                    os.mkdir(new_path)
                except:
                    pass
            blob.download_to_filename(local_dir + root_path + "/" + filename)  # Download
            extracted.append(local_dir + root_path + "/" + filename)
    return extracted


def remove_extracted_data_from_bucket(file_list,level=None):
    """ Delete all the files on the directories listed as well as directories themselves, based on the variable level of depth of           the directory, in the local storage of the VM in the notebook"""
    paths = []
    for file in file_list:
        paths.append(os.path.split(file)[0])
    unique_paths = list(set(paths))

    for path in unique_paths:
        try:
            shutil.rmtree(path) #delete all files in the directory
        except:
            pass
        if level!=None:
            if len(path.split('/'))-level>=4: #in order not delete "/home/jupyter/folder" directories
                   for i in range(level):
                        del_dir = path
                        del_dir = os.path.split(del_dir)[0]
                        try:
                            os.rmdir(del_dir)
                        except:
                            print("Deletion failed, there's still data on "+del_dir)
            else:
                   print('Too deep level of deletion!')
