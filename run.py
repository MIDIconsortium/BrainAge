import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import re
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import densenet169, densenet121, DenseNet  
import monai
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, 
    ToTensor,
    LoadNifti,
    ToTensor,
)

class T2_dataset(Dataset):
    """External T2 dataset dataset"""

    def __init__(self, csv_file, transform = None, return_metrics=False):
        self.file_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.return_metrics = return_metrics
    def __len__(self):
        return len(self.file_frame)

    def __getitem__(self, idx):
        stack_name = self.file_frame.iloc[idx]['file_name']
        tensor = self.transform(stack_name)  
        tensor = (tensor - tensor.mean())/tensor.std()
        tensor = torch.clamp(tensor,-3.5,3.5)
        ID = re.search('IXI[0-9]{1,}',stack_name).group(0)
        if self.return_metrics:
            age = self.file_frame.iloc[idx]['Age']  
            return tensor, age, ID
        else:
            return tensor, ID
    
def get_test_loader(csv_file,
                        batch_size=4):

    test_transforms = Compose([LoadNifti(image_only=True), ToTensor()])
   

    test_dataset = T2_dataset(csv_file, transform=test_transforms, return_metrics=return_metrics)
    

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader


def evaluate_with_age(net, data_loader, eval_criterion, device):
    val_running_loss = 0
    valid_count = 0 
    true_ages = []
    pred_ages = []
    ID2pred = {}
    ID2age = {}
    with torch.no_grad():
        net.eval()
        for k, data in enumerate(data_loader):
            t2, age, IDs = data
            t2 = t2.to(device=device, dtype = torch.float)
            age = age.to(device=device, dtype=torch.float)
            age = age.reshape(-1,1)

            pred_age = net(t2)
            for pred, true, ID in zip(pred_age, age, IDs):
                pred_ages.append(pred.item())
                true_ages.append(true.item())

                ID2pred[ID] = np.round(pred.item(), 1)
                ID2age[ID] = np.round(true.item(), 1)

                    
            
                
            val_running_loss += eval_criterion(pred_age, age).sum().detach().item()
            valid_count += t2.shape[0]
            
        val_loss = val_running_loss/valid_count
        corr_mat = np.corrcoef(true_ages, pred_ages)
        corr = corr_mat[0,1]

        return val_loss, corr, true_ages, pred_ages, ID2pred, ID2age
    
def evaluate_without_age(net, data_loader, device):
    ID2pred = {}
    with torch.no_grad():
        net.eval()
        for k, data in enumerate(data_loader):
            t2,  IDs = data
            t2 = t2.to(device=device, dtype = torch.float)

            pred_age = net(t2)
            for pred, ID in zip(pred_age, IDs):
                ID2pred[ID] = np.round(pred.item(), 1)

        return ID2pred

if __name__ == "__main__":
    net = DenseNet(3,1,1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--return_metrics', dest='return_metrics', action='store_true')
    parser.add_argument('--processed_csv_file', type=str,  default='./brain_age_evaluation_dataset.csv')
    
    parser.set_defaults(return_metrics=False)
    args = parser.parse_args()
    if args.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    net.load_state_dict(torch.load('./raw_T2.pt'))
    net = net.to(device)
    eval_criterion = nn.L1Loss(reduction='sum')
    if args.return_metrics:
        loader = get_IXI_test_loader(args.processed_csv_file, return_metrics=True)
        loss, corr, true_ages, pred_ages, ID2pred, ID2truth = evaluate_with_age(net, loader, eval_criterion, device)
        pd.DataFrame([ID2pred,ID2truth], index=['Predicted age (years)', 'True age (years)']).T.to_csv('./T2_brain_age_predictions.csv',index=False)
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.scatter(true_ages, pred_ages, alpha=0.3)
        ax.plot(true_ages, true_ages,linestyle= '--', color='black')
        ax.set_ylim([min(true_ages), max(true_ages)])
        ax.set_aspect('equal')
        ax.set_xlabel('Chronological age')
        ax.set_ylabel('Predicted age')
        ax.set_title('MAE = {:.2f} years, p = {:.2f}\n'.format(loss, corr))
        fig.savefig('./T2_scatter.png', facecolor='w')
    else:
        loader = get_IXI_test_loader(args.processed_csv_file)
        ID2pred = evaluate_without_age(net, loader, device)
        pd.DataFrame(ID2pred, index=['Predicted age (years)']).T.to_csv('./T2_brain_age_predictions.csv',index=False)
    


   
    
    
    









