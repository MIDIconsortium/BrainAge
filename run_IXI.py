import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import re
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

class IXI_dataset(Dataset):
    """t1 IoPPN dataset"""

    def __init__(self, csv_file, transform = None):
        self.file_frame = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.file_frame)

    def __getitem__(self, idx):
        stack_name = self.file_frame.iloc[idx]['file_name']
        tensor = self.transform(stack_name)  
        tensor = (tensor - tensor.mean())/tensor.std()
        tensor = torch.clamp(tensor,-3.5,3.5)
        age = self.file_frame.iloc[idx]['Age']  
        ID = re.search('IXI[0-9]{1,}',stack_name).group(0)
        return tensor, age, ID
    
def get_IXI_test_loader(csv_file,
                           batch_size=4,
                        flip=False):
    if flip:
        test_transforms = Compose([LoadNifti(image_only=True), RandFlip(prob=0.5, spatial_axis=0), ToTensor()])
    else:
        test_transforms = Compose([LoadNifti(image_only=True), ToTensor()])
   

    test_dataset = IXI_dataset(csv_file, transform=test_transforms)
    

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader

def get_IXI_test_loader(csv_file,
                           batch_size=4,
                        flip=False):

    test_transforms = Compose([LoadNifti(image_only=True), ToTensor()])
   

    test_dataset = IXI_dataset(csv_file, transform=test_transforms)
    

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader

def evaluate(net, data_loader, eval_criterion):
    val_running_loss = 0
    valid_count = 0 
    true_ages = []
    pred_ages = []
    ID2pred = {}
    ID2age = {}
    with torch.no_grad():
        net.eval()
        for k, data in enumerate(data_loader):
            t2, age, accs = data
            t2 = t2.to(device=device, dtype = torch.float)
            age = age.to(device=device, dtype=torch.float)
            age = age.reshape(-1,1)

            pred_age = net(t2)
            for pred, true, acc in zip(pred_age, age, accs):
                pred_ages.append(pred.item())
                true_ages.append(true.item())

                acc2pred[acc] = pred.item()
                acc2age[acc] = true.item()

                    
            
                
            val_running_loss += eval_criterion(pred_age, age).sum().detach().item()
            valid_count += t2.shape[0]
            
        val_loss = val_running_loss/valid_count
        corr_mat = np.corrcoef(true_ages, pred_ages)
        corr = corr_mat[0,1]

        return val_loss, corr, true_ages, pred_ages, ID2pred, ID2age

if __name__ == "__main__":
    net = DenseNet(3,1,1)
    device = torch.device('cuda:0')
    net = net.to(device)
    net.load_state_dict(torch.load(os.path.join(os.getcwd(),'train_test_pool_single_GPU.pt')))
    eval_criterion = nn.L1Loss(reduction='sum')
    loader = get_IXI_test_loader(os.path.join(os.getcwd(),'ixi_test_dataset.csv'))

    loss, corr, true_ages, pred_ages, acc2pred, acc2truth = evaluate(net, loader, eval_criterion)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.scatter(true_ages, pred_ages, alpha=0.3)
    ax.plot(true_ages, true_ages,linestyle= '--', color='black')
    ax.set_ylim([min(true_ages), max(true_ages)])
    ax.set_aspect('equal')
    ax.set_xlabel('Chronological age')
    ax.set_ylabel('Predicted age')
    ax.set_title('MAE = {:.2f} years, p = {:.2f}\n'.format(loss, corr))
    fig.savefig(os.path.join(os.getcwd(),'WOOD_IXI.png'), facecolor='w')








