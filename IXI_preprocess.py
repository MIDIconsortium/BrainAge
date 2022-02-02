import numpy as np
import nibabel as nib
import monai
from monai.transforms import (
    AddChannel, 
    Resize, 
    ScaleIntensity, 
    ToTensor,
    Randomizable,
    LoadNifti,
    Spacing,
    ResizeWithPadOrCrop
)
import matplotlib.pyplot as plt
import sys
import os
import re
import pandas as pd


def preprocess(input_path, save_path):
    border = 5
    min_dim = 130
    resize = Resize(spatial_size=(120, 120, 120), mode='trilinear')
    crop_pad = ResizeWithPadOrCrop(spatial_size=(180,180,180))
    ID = re.search('IXI[0-9]{3}',input_path).group(0)
    arr, _ = LoadNifti()(input_path)
    arr = AddChannel()(arr)
    arr = arr[:,:,::-1,:].copy()
    arr_resampled =  Spacing(pixdim=(1., 1., 1.), mode='bilinear')(arr,_['affine'])[0]

    if arr_resampled.shape[-1] > min_dim and arr_resampled.shape[-2] > min_dim and arr_resampled.shape[-3] > min_dim:
        mid_slice = arr_resampled.squeeze()[:,:,int(arr_resampled.shape[-1]/2)]
        mask = mid_slice>0.5*mid_slice.std()
        a, b = np.argmax(mask, axis=0)[int(mask.shape[1]/2)], np.argmax(mask, axis=1)[int(mask.shape[0]/2)]
        a1, b1 = np.argmax(np.flipud(mask), axis=0)[int(mask.shape[1]/2)], np.argmax(np.fliplr(mask), axis=1)[int(mask.shape[0]/2)]


        if a > border:
            a -= border
        else:
            a = 0
        if b > border:
            b -= border
        else:
            b = 0
        if b1 > border:
            b1 -= border
        else:
            b1 = 1
        if a1 > border:
            a1 -= border
        else:
            a1 = 1

        cropped = crop_pad(arr_resampled[:,a:-a1, b:-b1,:])
        resized = resize(cropped)

        new_image = nib.Nifti1Image(resized, affine=np.eye(4))

        nib.save(new_image, save_path)

if __name__ == "__main__":
    input_path, excel_path = sys.argv[1:]
    save_dir = os.path.join(os.getcwd(),'IXI_nii/')
    os.mkdir(save_dir)
    for root, dirs, files in os.walk(input_path):
        for f in files:
            nii_path = os.path.join(root, f)
            preprocess(nii_path, save_dir + f[:-3])
    df = pd.read_excel(excel_path)
    df = df[~df['AGE'].isnull()].reset_index(drop=True)
    df = df.drop_duplicates(subset='IXI_ID', keep=False).reset_index(drop=True)

    IDs = df['IXI_ID'].tolist()

    paths = []
    ages = []
    nii_path = os.path.join(os.getcwd(),'IXI_nii')
    for f in os.listdir(nii_path):
        ID = int(f[:-4])
        if ID not in IDs:
            continue
        row = df[df['IXI_ID'].astype(int)==ID]
        age = int(row['AGE'])
        paths.append(os.path.join(nii_path, f))
        ages.append(age)

    pd.DataFrame({'file_name':paths,'Age':ages}).to_csv(os.path.join(os.getcwd(),'IXI_test_dataset.csv'), index=False)













