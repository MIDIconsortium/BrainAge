import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import re
import os
import preprocess
import torch
import nibabel as nib
import tqdm

def plot_histogram(axis, arr, num_positions=50, label=None, alpha=0.05, color=None):
    values = arr.ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ixi_comp', dest='ixi', action='store_true')
    parser.set_defaults(ixi=False)
    parser.add_argument('--ixi_nii_dir', type=str, default='None')
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    args = parser.parse_args()
    
    if args.ixi:
        assert os.path.exists(args.ixi_nii_dir), 'path to IXI .nii files not valid'
        raw_df = pd.read_excel(args.csv_file)
        raw_df = raw_df[~raw_df['AGE'].isnull()].reset_index(drop=True)
        raw_df = raw_df.drop_duplicates(subset='IXI_ID', keep=False).reset_index(drop=True)
        IDs = raw_df['IXI_ID'].tolist()

        paths = []
        ages = []
        ids = []
        for root, dirs, files in os.walk(args.ixi_nii_dir):
            for f in files:
                ID = int(re.search('[0-9]{3}',f).group(0))
                if ID not in IDs:
                    continue
                row = raw_df[raw_df['IXI_ID'].astype(int)==ID]
                ages.append(np.round(row['AGE'].values[0], 1))
                paths.append(os.path.join(root, f))
                ids.append(ID)
        ixi_df = pd.DataFrame({'file_name':paths,'ID':ids, 'Age':ages})
        df = pd.read_csv(args.csv_file)

    else:
        df = pd.read_csv(args.csv_file)
    assert 'file_name' in df.columns, '''No column named 'file_name' in csv_file'''
    assert 'ID' in df.columns, '''No column named 'ID' in csv_file'''
    fig, ax = plt.subplots(dpi=100)
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        file_name = row['file_name']
        processed_arr = preprocess.preprocess(file_name)
        plot_histogram(ax, processed_arr, color='b', label='User data')
    if args.ixi:
        for index, row in tqdm.tqdm(ixi_df.iterrows(), total=df.shape[0]):
            file_name = row['file_name']
            processed_arr = preprocess.preprocess(file_name)
            plot_histogram(ax, processed_arr, color='r', label='IXI')
        
    
    
    ax.set_xlim(-100, 2000)
    ax.set_ylim(0, 0.004);
    ax.set_title('Original histograms of all samples')
    ax.set_xlabel('Intensity')
        

    fig.savefig('./{}_intensity_hist.png'.format(args.project_name), facecolor='w')
