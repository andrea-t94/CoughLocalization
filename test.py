from tf_features_extractor import spectrogram_extractor
from helpers import mono_to_color, open_fat_image
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img

audio = r"C:\Users\Administrator\Desktop\Voicemed\Annotation_Project\preview_headset\58.wav"

spectro = spectrogram_extractor(audio)
newImg = mono_to_color(spectro)
newImg = open_fat_image(newImg)
newImg.show()
newImg.save(out)
