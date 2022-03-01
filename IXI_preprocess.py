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
    ID = re.search('IXI[0-9]{3}',input_path).group(0)
    arr, _ = LoadNifti()(input_path)
    arr, affine, aff_trans, ornt_trans = reorder_voxels(arr, _['affine'], 'LPS')
    #arr = align_volume_to_ref(arr, _['affine'])
    arr = AddChannel()(arr)
    #arr = arr[:,:,::-1,:].copy()
    #arr_resampled =  Spacing(pixdim=(1., 1., 1.), mode='bilinear')(arr,_['affine'])[0]
    arr_resampled =  Spacing(pixdim=(1., 1., 1.), mode='bilinear')(arr,aff_trans)[0]

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
    else:
        print(input_path)

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
        paths.append(os.path.join(processed_nii_path, f))
        ages.append(age)

    pd.DataFrame({'file_name':paths,'Age':ages}).to_csv(os.path.join(os.getcwd(),'IXI_test_dataset.csv'), index=False)













