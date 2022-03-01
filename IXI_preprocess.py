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
import os
import re
import pandas as pd
import argparse

def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy()
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume


def preprocess(input_path, save_path):
    border = 5
    min_dim = 130
    resize = Resize(spatial_size=(120, 120, 120), mode='trilinear')
    crop_pad = ResizeWithPadOrCrop(spatial_size=(180,180,180))
    ID = re.search('IXI[0-9]{3}',input_path).group(0)
    arr, _ = LoadNifti()(input_path)
    arr = align_volume_to_ref(arr, _['affine'])
    arr = AddChannel()(arr)
    #arr = arr[:,:,::-1,:].copy()
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
    
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--input_nii_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--processed_nii_dir', type=str, default='./IXI_nii')
    args = parser.parse_args()
    input_path, excel_path, processed_nii_path = args.input_nii_dir, args.csv_path, args.processed_nii_dir
    os.mkdir(processed_nii_path)
    for root, dirs, files in os.walk(input_path):
        for f in files:
            nii_path = os.path.join(root, f)
            preprocess(nii_path, os.path.join(processed_nii_path, f[:-3]))
    df = pd.read_excel(excel_path)
    df = df[~df['AGE'].isnull()].reset_index(drop=True)
    df = df.drop_duplicates(subset='IXI_ID', keep=False).reset_index(drop=True)

    IDs = df['IXI_ID'].tolist()

    paths = []
    ages = []
    for f in os.listdir(processed_nii_path):
        ID = int(re.search('[0-9]{3}',f).group(0))
        if ID not in IDs:
            continue
        row = df[df['IXI_ID'].astype(int)==ID]
        age = np.round(row['AGE'].values[0], 1)
        paths.append(os.path.join(nii_path, f))
        ages.append(age)

    pd.DataFrame({'file_name':paths,'Age':ages}).to_csv(os.path.join(os.getcwd(),'IXI_test_dataset.csv'), index=False)













