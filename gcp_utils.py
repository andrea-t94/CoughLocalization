""" REPOSITORY OF UTILS FUNCTIONS """
""" COMPRISING FUNTIONS FOR COMMUNICATING WITH GCP ENVIRONMENT"""

import os
import shutil
from tqdm import tqdm
import multiprocessing
import warnings
from google.cloud import storage
import threading

######################################################################
#
#GCP INTERACTION FUNCTIONS
#
######################################################################


def file_upload_to_bucket(blob, file_path):
    blob.upload_from_filename(file_path)


def upload_to_bucket_v2(bucket_name, prefix, root_path='', file=None, local_dir=''):
    """ Upload file into a bucket, defined by bucket_name and prefix, the folder inside a bucket"""

    cores = multiprocessing.cpu_count()
    threads = []
    n = cores
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    upload_dir = local_dir + root_path

    if file != None:
        """Uploads a file to the bucket."""
        blob = bucket.blob(prefix + "/" + file)
        blob.upload_from_filename(upload_dir + "/" + file)

        print('File {} uploaded to {}.'.format(
            file,
            bucket_name + "/" + prefix))

    else:
        """Uploads a directory to the bucket."""
        for root, dirs, files in os.walk(upload_dir):

            for file in files:
                blob = bucket.blob(prefix + "/" + file)
                p = threading.Thread(target=file_upload_to_bucket, args=(blob, root + "/" + file))
                threads.append(p)

        print('Uploading to {}.'.format(bucket_name + "/" + prefix))
        with tqdm(total=len(threads)) as pbar:
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
                        warnings.warn(f"Low amount of files to process, lower than number of CPU cores, consisting of {n}",ResourceWarning)
                        n = len(threads)
                        pass
                for i in range(n):
                    threads[i].join()
                    pbar.update(1)

                threads = threads[n:]


def file_download_from_bucket(blob, file_path):
    blob.download_to_filename(file_path)


def extract_from_bucket_v2(bucket_name, prefix, root_path, local_dir='', file=None, max_samples=100000,
                           labels=['']):
    """ Download file from a bucket, identified by bucket_name, prefix stands for folder inside the bucket"""
    cores = multiprocessing.cpu_count()
    threads = []
    n = cores

    try:
        os.makedirs(local_dir + root_path + "/")
    except:
        print("Removing tmp directory")
        shutil.rmtree(local_dir + root_path + "/")
        os.makedirs(local_dir + root_path + "/")

    extracted = []
    blob_names = []
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    if file != None:
        """ Download a single file """
        blob = bucket.blob(prefix + '/' + file)
        filename = blob.name
        new_dir = os.path.split(filename)[0]
        new_path = root_path
        for path in new_dir.split('/'):
            new_path += '/' + path
            try:
                os.mkdir(local_dir + new_path)
            except:
                pass
        blob.download_to_filename(local_dir + root_path + "/" + filename)
        #print(local_dir + filename)
        extracted.append(local_dir + root_path + "/" + filename)
        return extracted

    else:
        """ Download an entire folder """
        for i, label in enumerate(labels):
            blobs = bucket.list_blobs(prefix=prefix + "/" + label, max_results=max_samples)  # Get list of files
            for blob in blobs:
                filename = blob.name
                blob_names.append(filename)
                new_dir = os.path.split(filename)[0]
                new_path = root_path
                for path in new_dir.split('/'):
                    new_path += '/' + path
                    try:
                        os.mkdir(new_path)
                    except:
                        pass
                p = threading.Thread(target=file_download_from_bucket,
                                     args=(blob, local_dir + root_path + "/" + filename))
                threads.append(p)
                extracted.append(local_dir + root_path + "/" + filename)

        with tqdm(total=len(threads)) as pbar:
            while len(threads) > 0:
                if len(threads) < n:
                    n = len(threads)
                    warnings.warn(
                        f"Low amount of files to process, lower than number of CPU cores, consisting of {n}",
                        ResourceWarning)
                for i in range(n):
                    threads[i].start()
                for i in range(n):
                    threads[i].join()
                    pbar.update(1)

                threads = threads[n:]

        return (extracted, blob_names)

