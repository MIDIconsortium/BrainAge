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
import PreProcess

if __name__ == "__main__":
    if not os.path.exists('./processed_imgs/'):
        os.mkdir('./processed_imgs/')
    parser = argparse.ArgumentParser()
    parser.add_argument('--nii_path', nargs='*')
    parser.add_argument('--skull_strip', dest='skull_strip', action='store_true')
    parser.set_defaults(skull_strip=False)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--sequence', type=str, default='t2')
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--register', dest='register', action='store_true')
    parser.set_defaults(register=False)
    
    
    
    args = parser.parse_args()
    for path in args.nii_path:
        processed_arr = PreProcess.preprocess(input_path=path, use_gpu=args.gpu, save_dir=None, skull_strip=args.skull_strip, register=args.register, project_name=args.project_name)
        if not type(processed_arr)==np.ndarray:
            continue
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,12))
        ax1.imshow(np.rot90(processed_arr.squeeze()[65,:,:]), cmap='gray')
        ax2.imshow(processed_arr.squeeze()[:,:,65], cmap='gray')
        fig.savefig('./processed_imgs/{}.png'.format(len(os.listdir('./processed_imgs/'))))
        plt.close()
