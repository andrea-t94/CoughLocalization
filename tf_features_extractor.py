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


def spectrogram_extractor(params, amplitude_spectro=False):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    params: An instance of Params containing hyperparameters.

  Returns:
    A model accepting (num_samples,) waveform input and emitting:
    - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
  """
  waveform = layers.Input(batch_shape=(None,), dtype=tf.float32)
  waveform_padded = features_lib.pad_waveform(waveform, params)
  log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
    waveform_padded, params, amplitude_spectro)
  frames_model = Model(
    name='spectro_extractor', inputs=waveform,
    outputs=[log_mel_spectrogram, features])

  return frames_model

@ray.remote
class Annotator(object):
  def __init__(self, params, i):
    if sys.platform == 'linux':
      psutil.Process().cpu_affinity([i])
    # Load the extractor and initialize annotations params and variables.
    self.params = params
    self.extractor = spectrogram_extractor(self.params, amplitude_spectro=True)

  def spectrogram_image_tf(self, audio, out, binary=False):
    '''based on tf, convert audio in logmel-spectrogram of dimension (N_MELS, N_SPECTRO) and subsequently into immage of dimension (N_MELS, N_SPECTRO) pixels'''
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
    spectrogram, features = self.extractor(waveform)

    # spectrogram reshaping
    spectrogram = tf.transpose(spectrogram)
    image_shape = spectrogram.get_shape().as_list()
    newImg = mono_to_color(spectrogram)
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
    len_signal, spectro_shape = self.spectrogram_image_tf(audio, out, False)

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
