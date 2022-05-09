import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import re
import argparse
import os
import pre_process
from monai.networks.nets import DenseNet  
import torch
import nibabel as nib
import tqdm
import datetime
from collections import OrderedDict
import time

def convert_state_dict(input_path):
    #function to remove the keywork 'module' from pytorch state_dict (which occurs when model is trained using nn.DataParallel)
    new_state_dict = OrderedDict()
    state_dict = torch.load(input_path, map_location='cpu')
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict
           
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--return_metrics', dest='return_metrics', action='store_true')
    parser.set_defaults(return_metrics=False)
    parser.add_argument('--skull_strip', dest='skull_strip', action='store_true')
    parser.set_defaults(skull_strip=False)
    parser.add_argument('--pred_correction', dest='pred_correction', action='store_true')
    parser.set_defaults(pred_correction=False)
    parser.add_argument('--ensemble', dest='ensemble', action='store_true')
    parser.set_defaults(ensemble=False)
    parser.add_argument('--sequence', type=str, default='t2')
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    args = parser.parse_args()
    
    if not os.path.exists('./{}'.format(args.project_name)):
        os.mkdir('./{}'.format(args.project_name))
    else:
        raise ValueError('project name {} aready used'.format(args.project_name))
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')     
    
    if args.sequence == 't2':
        if args.skull_strip:
            state_dict = convert_state_dict('./Models/T2/Skull_stripped/seed_42.pt')
            net = DenseNet(3,1,1)
            net.load_state_dict(state_dict)
            net = net.to(device)
            net.eval()
        else:
            state_dict = convert_state_dict('./Models/T2/Raw/seed_42.pt')
            net = DenseNet(3,1,1)
            net.load_state_dict(state_dict)
            net = net.to(device)
            net.eval()
    elif args.sequence == 't1':
        if args.skull_strip:
            if args.ensemble:
                net = []
                for path in os.listdir('./Models/T1/Skull_stripped/'):
                    state_dict = convert_state_dict('./Models/T1/Skull_stripped/' + path)
                    Net = DenseNet(3,1,1)
                    Net.load_state_dict(state_dict)
                    Net = Net.to(device)
                    Net.eval()
                    net.append(Net)
            else:
                state_dict = convert_state_dict('./Models/T1/Skull_stripped/seed_60.pt')
                net = DenseNet(3,1,1)
                net.load_state_dict(state_dict)
                net = net.to(device)
                net.eval()        
        else:
            raise ValueError('Raw T1 model not currently handled. Please specify --skull_strip if skull-stripped and registered (MNI152) T1 model is desired')
    else:
        raise ValueError('{} MRI sequence not currently handled (must be one of t2 or t1; DWI and FLAIR coming soon!)'.format(args.sequence))

    df = pd.read_csv(args.csv_file)
    
    assert args.sequence in ['t1','t2'], '''Unsupported sequence provided ({})'''.format(args.sequence)
    
    assert 'file_name' in df.columns, '''No column named 'file_name' in csv_file'''
    
    assert 'ID' in df.columns, '''No column named 'ID' in csv_file'''
    
    if args.return_metrics:
        assert 'Age' in df.columns, '''No column named 'Age' in csv_file, can't return brain-age metrics (MAE, Pearson's etc.)'''  
        
    if args.pred_correction:
        assert 'Age' in df.columns, '''No column named 'Age' in csv_file, can't correct for bias in brain-age predictions'''
        
    brain_predicted_ages = []
    chronological_ages = []
    IDs = []
    # Evaluation loop
    with torch.no_grad():
        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            file_name = row['file_name']
            ID = row['ID'] 
            if args.return_metrics:
                age = row['Age']
            processed_arr = pre_process.preprocess(input_path=file_name, use_gpu=args.gpu, skull_strip=args.skull_strip, register=args.sequence=='t1', project_name=args.project_name)
            if not type(processed_arr)==np.ndarray:
                continue
            tensor = torch.from_numpy(processed_arr).view(1,1,130,130,130)
            tensor = (tensor - tensor.mean())/tensor.std()
            tensor = torch.clamp(tensor,-1,5)
            tensor = tensor.to(device=device, dtype = torch.float)
            if args.sequence=='t1':
                if args.ensemble:
                    temp_preds = []
                    for Net in net:
                        temp_preds.append(np.round(Net(tensor).detach().cpu().item(), 1))
                    brain_predicted_ages.append(np.mean(temp_preds))
                else:
                     brain_predicted_ages.append(np.round(net(tensor).detach().cpu().item(), 1))
            else:
                if args.pred_correction and not args.skull_strip:              
                    brain_predicted_ages.append(np.round(net(tensor).detach().cpu().item(), 1) - (-0.0627*age + 2.54))
                elif args.pred_correction and args.skull_strip:              
                    brain_predicted_ages.append(np.round(net(tensor).detach().cpu().item(), 1) - (-0.0854*age + 2.67))
                else:
                    brain_predicted_ages.append(np.round(net(tensor).detach().cpu().item(), 1))
            if args.return_metrics:
                chronological_ages.append(np.round(row['Age'],1))
            IDs.append(ID)
            
    if args.return_metrics:
        
        pd.DataFrame({'ID':IDs,'Chronological age':chronological_ages,'Predicted_age (years)':brain_predicted_ages}).set_index('ID').to_csv('./{}_brain_age_output.csv'.format(args.project_name))
        
        MAE = sum([np.abs(a-b) for a, b in zip(brain_predicted_ages, chronological_ages)])/len(brain_predicted_ages)
        corr_mat = np.corrcoef(chronological_ages, brain_predicted_ages)
        corr = corr_mat[0,1]

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.scatter(chronological_ages, brain_predicted_ages, alpha=0.3)
        ax.plot(chronological_ages, chronological_ages,linestyle= '--', color='black')
        ax.set_ylim([min(chronological_ages), max(chronological_ages)])
        ax.set_aspect('equal')
        ax.set_xlabel('Chronological age')
        ax.set_ylabel('Predicted age')
        ax.set_title('MAE = {:.2f} years, p = {:.2f}\n'.format(MAE, corr))
        fig.savefig('./{}/scatter.png'.format(args.project_name), facecolor='w')
        
    else:
        pd.DataFrame({'ID':IDs,'Predicted_age (years)':brain_predicted_ages}).set_index('ID').to_csv('./{}/brain_age_output.csv'.format(args.project_name)) 
