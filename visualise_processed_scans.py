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
import t1_skull_strip_register_and_preprocess
import preprocess

if __name__ == "__main__":
    if not os.path.exists('./processed_imgs/'):
        os.mkdir('./processed_imgs/')
    parser = argparse.ArgumentParser()
    parser.add_argument('--nii_path', nargs='*')
    parser.add_argument('--skull_strip', dest='skull_strip', action='store_true')
    parser.set_defaults(skull_strip=False)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(skull_strip=False)
    parser.add_argument('--sequence', type=str, default='t2')
    args = parser.parse_args()
    for path in args.nii_path:
        if args.sequence == 't2' and not args.skull_strip:
            processed_arr = preprocess.preprocess(path)
        elif args.sequence == 't2' and args.skull_strip:
            processed_arr = t2_skull_strip_and_preprocess.preprocess(path, args.gpu)
        elif args.sequence == 't1' and args.skull_strip:
            t1_skull_strip_register_and_preprocess.preprocess(path, args.gpu)
        else:
            raise ValueError('MRI sequence {} (skull_strip: {}) not currently handled'.format(args.sequence, args.skull_strip))
        if not type(processed_arr)==np.ndarray:
            continue
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,12))
        ax1.imshow(np.rot90(processed_arr.squeeze()[65,:,:]), cmap='gray')
        ax2.imshow(arr.squeeze()[:,:,65], cmap='gray')
        fig.savefig('./processed_imgs/{}.png'.format(len(os.listdir('./processed_imgs/'))))
        plt.close()
