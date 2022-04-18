import numpy as np
import nibabel as nib
import monai
from monai.transforms import (
    AddChannel, 
    Resize,
    Spacing,
    ResizeWithPadOrCrop
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import t2_skull_strip_and_preprocess
if __name__ == "__main__":
    if not os.path.exists('./processed_imgs/'):
        os.mkdir('./processed_imgs/')
    parser = argparse.ArgumentParser()
    parser.add_argument('--nii_path', nargs='*')
    args = parser.parse_args()
    for path in args.nii_path:
        print(type(path))
        arr = t2_skull_strip_and_preprocess.preprocess(args.nii_path)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,12))
        ax1.imshow(arr.squeeze()[65,:,:], cmap='gray')
        ax2.imshow(arr.squeeze()[:,:,65], cmap='gray')
        fig.savefig('./processed_imgs/{}.png'.format(len(os.listdir('./processed_imgs/'))))
        plt.close()
