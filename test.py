import os
import glob
from pathlib import Path

path = '/Users/andreatamburri/Desktop/tmp/covidClass/v_2/crop_samples'
test_path = 'COUGHVIDAnnotated/covidClass/v_2/crop_samples'
tmp_dirs = []
for root, dirs, files in os.walk(path):
    print(os.path.commonpath([os.path.relpath(root),test_path]))
    for dir in dirs:
        if dir not in test_path:
            tmp_dirs.append(dir)
    #for file in files:
     #  print(file)