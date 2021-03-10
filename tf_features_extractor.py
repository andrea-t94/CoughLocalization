# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Core model definition of YAMNet."""
import sys
import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras import Model, layers
from audio_utils import mono_to_color, open_fat_image
import tf_features as features_lib
import soundfile as sf
import resampy
import uuid
from datetime import datetime
from typing import Dict, List
import os
import shutil


def features_extractor(params, amplitude_spectro=True):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    params: An instance of Params containing hyperparameters.

  Returns:
    A model accepting (num_samples,) waveform input and emitting:
    - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
    - mfcc: (num_spectrogram_frames, n_mfccs <= num_mel_bins) mfcc feature matrix
  """
  waveform = layers.Input(batch_shape=(None,), dtype=tf.float32)
  waveform_padded = features_lib.pad_waveform(waveform, params)
  log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
    waveform_padded, params, amplitude_spectro)
  mfcc = tf.signal.mfccs_from_log_mel_spectrograms(
    features)[..., :params.n_mfcc]
  frames_model = Model(
    name='spectro_extractor', inputs=waveform,
    outputs=[log_mel_spectrogram, mfcc, features])  #mfcss already put in (n*96*14)

  return frames_model


class Extractor(object):
  ''' the extractor compute featurse such as spectrograms and mfccs and creates images out of them'''
  def __init__(self, params, feat_list, i):
    if sys.platform == 'linux':
      psutil.Process().cpu_affinity([i])
    # Load the extractor and initialize annotations params and variables.
    self.params = params
    self.feat_list = feat_list
    self.extractor = features_extractor(self.params)


  def feature_tf(self, audio):
    '''
    based on tf, convert audio in features of dimension (N_BINS, N_SPECTRO)
    '''
    # Decode the WAV file.
    signal, sr = sf.read(audio, dtype=np.int16)
    assert signal.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = signal / 37768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != self.params.sample_rate:
      waveform = resampy.resample(waveform, sr, self.params.sample_rate)
      len_signal = len(waveform)
    spectrogram, mfcc, features = self.extractor(waveform)

    return spectrogram, mfcc, features

  def feature_image_tf(self, audio, out, binary=False):
    '''
    based on tf, convert audio in features of dimension (N_BINS, N_SPECTRO) and subsequently into image
    of dimension (N_BINS, N_SPECTRO) pixels
    It takes in input a list of features, currently we have:
    - spectrogram
    - mfcc
    '''
    # Decode the WAV file.
    signal, sr = sf.read(audio, dtype=np.int16)
    assert signal.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = signal / 37768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != self.params.sample_rate:
      waveform = resampy.resample(waveform, sr, self.params.sample_rate)
      len_signal = len(waveform)
    spectrogram, mfcc, features = self.extractor(waveform)
    #dict for mapping features calculated by extractor
    feat_dict = {'spectrogram': spectrogram,
                 'mfcc': mfcc}
    calc_features = [key for key,val in feat_dict.items()]

    final_list = set(feat_list).intersection(set(calc_features))
    if len(final_list) < len(feat_list):
      raise Warning(f"the list of features to extract contains feature not implemented in the code base! "
                    f"features calculated are {final_list}")

    for feat_name in final_list:
      # feature reshaping
      feat = feat_dict[feat_name]
      feat = tf.transpose(feat)
      image_shape = feat.get_shape().as_list()
      newImg = mono_to_color(feat)
      if not binary:
        newImg = open_fat_image(newImg)
        newImg.save(out)
      else:
        np_img = np.array(newImg).flatten()  # flattened image (options available)
        f = open(f'{out}.bin', "wb")
        mydata = np_img
        myfmt = f'{len(mydata)}B'
        bin = struct.pack(myfmt, *mydata)
        f.write(bin)
        f.close()

    return len_signal, image_shape



@ray.remote
class Annotator(Extractor):
  ''' based on the Extractor core, builds COCO dataset with annotations'''

  def __init__(self, params, features, i):
    Extractor.__init__(self, params, features, i)

  #fileInfo : (gcp_output_uri, local_path, gcp_path, (xmin, xmax))
  def annotation_factory(
          self,
          fileName: str,
          fileInfo: tuple,
          image_path: str,
  ):
    file_images = []
    file_annotations = []
    audio = fileInfo[1] + ".wav"
    annotation_events = fileInfo[-1]

    # convert to PNG
    fileNameOut = f"{fileName}.png"
    out = fr"{image_path}/{fileNameOut}"
    len_signal, spectro_shape = self.feature_image_tf(audio, out, False)

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
    file_images.append(image)

    for event in annotation_events:
      starting_event, ending_event = event[0], event[-1]
      starting_point = starting_event * self.params.sample_rate / frames_per_spectro  # events in seconds
      ending_point = ending_event * self.params.sample_rate / frames_per_spectro
      annotation = {
        "iscrowd": 0,
        "image_id": imageUuid,
        "bbox": [starting_point, N_MELS, ending_point - starting_point, N_MELS],
        # top left x & y position, width and height
        "category_id": 0,
        "id": uuid.uuid4().hex
      }
      file_annotations.append(annotation)

    return file_images, file_annotations


@ray.remote
class Cropper(Extractor):
  '''
   based on the Extractor core, crop audio files according to annotations
   and return a dictionary as ##cropFilePath :(fileName, labelName, label)
  '''

  def __init__(self, params, features, i):
    Extractor.__init__(self, params, features, i)

  # fileInfo : (gcp_output_uri, local_path, gcp_path, (xmin, xmax))
  def cropping_factory(
          self,
          fileName: str,
          fileInfo: tuple,
          crop_path: str,
          labelMapDict: Dict,
          mfcc_path: str,
          spectro_path: str,
  ):
    defaultDict = {}
    listMfcc = []
    listSpectro = []
    # info retrieval
    filePath = f'{fileInfo[1]}.wav'
    gcp_output_uri = fileInfo[0]
    annotations = fileInfo[-1]
    for key, val in labelMapDict.items():
      if key in filePath:
        label = val
        labelName = key
    for path in [mfcc_path, spectro_path, crop_path]:
      try:
        os.makedirs(f"{path}/{labelName}")
      except:
        pass

    # Decode the WAV file and crop it.
    signal, sr = sf.read(filePath, dtype=np.int16)
    for i, annotation in enumerate(annotations):
      starting_point = int(annotation[0] * sr)
      ending_point = int(annotation[1] * sr) + 1
      cropFileName = f"{fileName}_{i}"
      crop_audio_path = f'{crop_path}/{labelName}/{cropFileName}.wav'
      signal1, sr = sf.read(filePath, dtype=np.int16, start=starting_point, stop=ending_point)
      if len(signal1) == 0:
        pass
      else:
        try:
          sf.write(crop_audio_path, signal1, sr)
        except:
          pass

        #Feature extraction and saving, the cropper for covid uses sliding windows spectros and mfccs
        spectrogram, mfcc, features = self.feature_tf(crop_audio_path)
        np_mfcc = np.array([mfcc, label])
        np_spectrogram = np.array([features, label])
        try:
          np.save(f"{mfcc_path}/{labelName}/{cropFileName}.npy", np_mfcc)
        except:
          pass
        try:
          np.save(f"{spectro_path}/{labelName}/{cropFileName}.npy", np_spectrogram)
        except:
          pass

      listMfcc.append(np_mfcc)
      listSpectro.append(np_spectrogram)

    return listMfcc, listSpectro
