# Cough Processing
Partial code based on my work in [Voicemed](https://www.voicemed.io/) API, an API built for Covid detection, as creator of CE Marked Health device in Italy.

Repo about all the techniques used for:
- audio signal processing
- cough annotation based on COCO Dataset
- Deep learning models for cough localization (based on YOLO-tf-v4, see my fork [repo](https://github.com/andrea-t94/tensorflow-yolov4-tflite))

The repository is mainly composed by:
- utils, for GCP interaction as well as audio processing techinques (e.g. spectrograms)
- tensorflow implementation of spectrogram extractor (from Yamnet), with simble "tf"
- annotators that work within a GCP environment, with tf and librosa audio preprocessing

Develop:
- easier approach to cough localization based on standard CNN, with spectro/mfcc as input.
- the main benefit are the faster data preprocessing and model inference (10X params less), very crucial in UX (difference in seconds to inference)
