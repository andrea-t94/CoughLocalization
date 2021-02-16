from tf_features_extractor import spectrogram_extractor
from helpers import mono_to_color, open_fat_image, spectrogram_image
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img
import params as spectro_params
import librosa
import tensorflow as tf
import soundfile as sf
import numpy as np
import resampy


def spectrogram_image_tf(audio, out):
    params = spectro_params.Params()
    #signal, sr = librosa.load(audio, sr=params.sample_rate)
    # Decode the WAV file.
    signal, sr = sf.read(audio, dtype=np.int16)
    assert signal.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = signal / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)
    extractor = spectrogram_extractor(params, amplitude_spectro=True)
    spectrogram, features = extractor(waveform)

    #spectrogram reshaping
    spectrogram = tf.transpose(spectrogram)
    image_shape = spectrogram.get_shape().as_list()
    newImg = mono_to_color(spectrogram)
    newImg = open_fat_image(newImg)
    newImg.save(out)

    return image_shape


audio = r"C:\Users\Administrator\Desktop\Voicemed\Annotation_Project\preview_headset\58.wav"
out = r"C:\Users\Administrator\Desktop\test_tf_spectro.png"
shape = spectrogram_image_tf(audio,out)
print(shape)