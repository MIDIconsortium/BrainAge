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

    parser = argparse.ArgumentParser()
    parser.add_argument('--nii_path', nargs='*')
    args = parser.parse_args()
    for path in parser.nii_path:
        print(path)
    #preprocess(args.nii_path)
