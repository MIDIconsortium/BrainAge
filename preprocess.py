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
def reorder_voxels(vox_array, affine, voxel_order):
    '''Reorder the given voxel array and corresponding affine.
    Parameters
    ----------
    vox_array : array
        The array of voxel data
    affine : array
        The affine for mapping voxel indices to Nifti patient space
    voxel_order : str
        A three character code specifing the desired ending point for rows,
        columns, and slices in terms of the orthogonal axes of patient space:
        (l)eft, (r)ight, (a)nterior, (p)osterior, (s)uperior, and (i)nferior.
    Returns
    -------
    out_vox : array
        An updated view of vox_array.
    out_aff : array
        A new array with the updated affine
    reorient_transform : array
        The transform used to update the affine.
    ornt_trans : tuple
        The orientation transform used to update the orientation.
    '''
    #Check if voxel_order is valid
    voxel_order = voxel_order.upper()
    if len(voxel_order) != 3:
        raise ValueError('The voxel_order must contain three characters')
    dcm_axes = ['LR', 'AP', 'SI']
    for char in voxel_order:
        if not char in 'LRAPSI':
            raise ValueError('The characters in voxel_order must be one '
                             'of: L,R,A,P,I,S')
        for idx, axis in enumerate(dcm_axes):
            if char in axis:
                del dcm_axes[idx]
    if len(dcm_axes) != 0:
        raise ValueError('No character in voxel_order corresponding to '
                         'axes: %s' % dcm_axes)

    #Check the vox_array and affine have correct shape/size
    if len(vox_array.shape) < 3:
        raise ValueError('The vox_array must be at least three dimensional')
    if affine.shape != (4, 4):
        raise ValueError('The affine must be 4x4')

    #Pull the current index directions from the affine
    orig_ornt = nib.io_orientation(affine)
    new_ornt = nib.orientations.axcodes2ornt(voxel_order)
    ornt_trans = nib.orientations.ornt_transform(orig_ornt, new_ornt)
    orig_shape = vox_array.shape
    vox_array = nib.apply_orientation(vox_array, ornt_trans)
    aff_trans = nib.orientations.inv_ornt_aff(ornt_trans, orig_shape)
    affine = np.dot(affine, aff_trans)

    return (vox_array, affine, aff_trans, ornt_trans)

def preprocess(input_path, save_path):
    border = 5
    min_dim = 130
    resize = Resize(spatial_size=(120, 120, 120), mode='trilinear')
    crop_pad = ResizeWithPadOrCrop(spatial_size=(180,180,180))
    arr, _ = LoadNifti()(input_path)
    arr, affine, aff_trans, ornt_trans = reorder_voxels(arr, _['affine'], 'LPS')
    arr = AddChannel()(arr)
    arr_resampled =  Spacing(pixdim=(1., 1., 1.), mode='bilinear')(arr, _['affine'])[0]
    print(aff_trans)
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
        
        return True
    else:
        #print('failed')
        return False

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--raw_csv_path', type=str, required=True)
    parser.add_argument('--processed_nii_dir', type=str, default='./T2_processed_nii/')
    parser.add_argument('--processed_csv_path', type=str, default='./brain_age_evaluation_dataset.csv')
    args = parser.parse_args()
    csv_path, eval_csv_path, save_dir = args.raw_csv_path, args.processed_csv_path, args.processed_nii_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    df = pd.read_csv(csv_path)
    df['processed_file_name'] = -1
    for i, row in df.iterrows():
        ID = row['ID']
        nii_path = row['file_name']
        out = preprocess(nii_path, os.path.join(save_dir, ID + '.nii'))
        if out:
            df.loc[i, 'processed_file_name'] = os.path.join(save_dir, ID + '.nii')
    df = df[df['processed_file_name']!=-1].reset_index(drop=True)
    df.to_csv(eval_csv_path, index=False)














