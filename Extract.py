import PIL
import pydicom
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from fastai.core import parallel

print("PyTorch version: ", torch.__version__)

window_center = 40
window_width = 80

paths = Path('/media/docear/My Passport/Kaggle/Hemorrhage/stage_1_train_images')
pathsTest = Path('/media/docear/My Passport/Kaggle/Hemorrhage/stage_1_test_images')
path = list(paths.iterdir())[0]
im = pydicom.read_file(str(path))

def window_and_normalize(im):
    rescaled = im.pixel_array * float(im.RescaleSlope) + float(im.RescaleIntercept)
    windowed = rescaled.clip(min=window_center-window_width, max=window_center+window_width)

    return (windowed + np.negative(window_center-window_width)) / (window_width * 2 * 1/255)

plt.imshow(window_and_normalize(im), cmap=plt.cm.bone)

def resize(src, dst, sz):
    im = pydicom.read_file(str(src))
    ary = window_and_normalize(im)
    im = PIL.Image.fromarray(ary.astype(np.int8), mode='L')
    im.resize((sz,sz), resample=PIL.Image.BICUBIC).save(f'{dst}/{src.stem}.png')
    
print('Processing Train Set')
def resize_112(path, _): resize(path, '/media/docear/My Passport/Kaggle/Hemorrhage/data/112/train', 112)
parallel(resize_112, list(paths.iterdir()), max_workers=12)

print('Processing Test Set')
def resize_112_test(path, _): resize(path, '/media/docear/My Passport/Kaggle/Hemorrhage/data/112/test', 112)
parallel(resize_112_test, list(pathsTest.iterdir()), max_workers=12)
