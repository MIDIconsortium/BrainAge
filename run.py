import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import re
import argparse
import os
import preprocess
from monai.networks.nets import DenseNet  
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--ixi', dest='ixi', action='store_true')
    parser.set_defaults(ixi=False)
    parser.add_argument('--return_metrics', dest='return_metrics', action='store_true')
    parser.set_defaults(return_metrics=False)
    parser.add_argument('--skull_strip', dest='skull_strip', action='store_true')
    parser.set_defaults(skull_strip=False)
    parser.add_argument('--sequence', type=str, default='t2')
    parser.add_argument('--ixi_nii_dir', type=str, default='None')
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    args = parser.parse_args()
    
    net = DenseNet(3,1,1)
    if args.sequence == 't2':
        if not args.skull_strip:
            net.load_state_dict(torch.load('./raw_T2.pt'))
        else:
            raise ValueError('Skull stripping not currently handled')
    else:
        raise ValueError('MRI sequence {} not currently handled'.format(args.sequence))
    if args.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    net = net.to(device)

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
        df = pd.DataFrame({'file_name':paths,'ID':ids, 'Age':ages})

    else:
        df = pd.read_csv(args.csv_file)
    assert 'file_name' in df.columns, '''No column named 'file_name' in csv_file'''
    assert 'ID' in df.columns, '''No column named 'ID' in csv_file'''
    if args.return_metrics:
        assert 'Age' in df.columns, '''No column named 'Age' in csv_file, can't return brain-age metrics (MAE, Pearson's etc.)'''
    brain_predicted_ages = []
    chronological_ages = []
    IDs = []
    # Evaluation loop
    net.eval()
    with torch.no_grad():
        for i, row in df.iterrows():
            file_name = row['file_name']
            nii = nib.load(file_name)
            processed_arr = preprocess.preprocess(file_name)
            tensor = torch.from_numpy(preprocessed_arr).view(1,1,120,120,120)
            tensor = (tensor - tensor.mean())/tensor.std()
            tensor = torch.clamp(tensor,-3.5,3.5)
            tensor = tensor.to(device=device, dtype = torch.float)
            ID = self.file_frame.iloc[idx]['ID']               

            brain_predicted_ages.append(np.round(net(tensor).item(), 1))
            if args.return_metrics:
                chronological_ages.append(np.round(row['Age'],1))
            IDs.append(ID)
    if args.return_metrics:
        out_df = pd.DataFrame({'ID':IDs,'Chronological age':chronological_ages,'Predicted_age (years)':brain_predicted_ages}).set_index('ID')
    else:
        out_df = pd.DataFrame({'ID':IDs,'Predicted_age (years)':brain_predicted_ages}).set_index('ID')
    if ages.ixi:
        out_df.to_csv('./IXI_output.csv')
    else:
        out_df.to_csv('./{}_output.csv'.format(args.project_name))
    
    if args.return_metrics:
        val_loss = sum([np.abs(a-b) for a, b in zip(brain_predicted_ages, chronological_ages)])/len(brain_predicted_ages)
        corr_mat = np.corrcoef(chronological_ages, brain_predicted_ages)
        corr = corr_mat[0,1]

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.scatter(true_ages, pred_ages, alpha=0.3)
        ax.plot(true_ages, true_ages,linestyle= '--', color='black')
        ax.set_ylim([min(true_ages), max(true_ages)])
        ax.set_aspect('equal')
        ax.set_xlabel('Chronological age')
        ax.set_ylabel('Predicted age')
        ax.set_title('MAE = {:.2f} years, p = {:.2f}\n'.format(loss, corr))
        fig.savefig('./{}_scatter.png'.format(args.project_name), facecolor='w')

    


   
    
    
    










