import numpy as np
import pandas as pd
import warnings
import time
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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from monai.networks.nets import DenseNet  

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from monai.transforms import (
    AddChannel, 
    Compose,
    Resize, 
    ScaleIntensity, 
    ToTensor,
    Randomizable,
    LoadNifti,
    Spacing,
    ResizeWithPadOrCrop,
    RandFlip, 
)

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

def train(net, optimizer, scheduler, train_loader, valid_loader, criterion, eval_criterion, save_path, epochs = 30, patience = 5):
    best_loss = 1e9
    num_bad_epochs = 0
    print('**BEGINNING TRAINING***')
    for epoch in range(epochs):
        start = time.time()
        train_loss = 0  
        net.train()
        train_count = 0
        if num_bad_epochs >= patience:
            return None
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            im, age = data
            im = im.to(device=device, dtype = torch.float)
            age = age.to(device=device, dtype=torch.float)
            age = age.reshape(-1,1)


            optimizer.zero_grad()
            pred_age = net(im)
            loss = criterion(pred_age, age)

            loss.backward()
            train_count += im.shape[0]

            train_loss += eval_criterion(pred_age, age).sum().detach().item()

            optimizer.step()

            
        train_loss/= train_count     
        val_loss, corr, *_ = evaluate(net, valid_loader, eval_criterion)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), save_path)
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1   
        
        end = time.time()
        lr = optimizer.param_groups[0]['lr']
        print('Epoch: {}, lr: {:.2E}, train loss: {:.1f}, valid loss: {:.1f}, corr: {:.2f}, best loss {:.1f}, number of epochs without improvement: {}'.format(epoch,
             lr, train_loss, val_loss, corr, best_loss, num_bad_epochs))

    return None
  
def evaluate(net, data_loader, eval_criterion):
  val_running_loss = 0
  valid_count = 0 
  true_ages = []
  pred_ages = []
  with torch.no_grad():
      net.eval()
      for k, data in enumerate(tqdm.tqdm(data_loader)):
          im, age = data
          im = im.to(device=device, dtype = torch.float)
          age = age.to(device=device, dtype=torch.float)
          age = age.reshape(-1,1)

          pred_age = net(im)
          for pred, chron_age in zip(pred_age, age):
              pred_ages.append(pred.item())
              true_ages.append(chron_age.item())

          val_running_loss += eval_criterion(pred_age, age).sum().detach().item()
          valid_count += im.shape[0]

      val_loss = val_running_loss/valid_count
      corr_mat = np.corrcoef(true_ages, pred_ages)
      corr = corr_mat[0,1]

      return val_loss, corr, true_ages, pred_ages
    
def process(csv_file, project_name, sequence, save_dir, skull_strip=False):
    df = pd.read_csv(csv_file)
    df['processed_file_name'] = -1
    print('***PRE-PROCESSING RAW NIFTI FILES***')
    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        file_path = row['file_name']
        ID = str(row['ID'])
        save_path = os.path.join(save_dir + 'processed_nii', ID + '.nii.gz')
        _ = pre_process.preprocess(input_path=file_path, save_path = save_path, use_gpu=args.gpu, skull_strip=args.skull_strip, register=args.sequence=='t1', project_name=args.project_name)
        df.loc[i, 'processed_file_name'] = save_path
    df.to_csv(save_dir + 'fine_tuning_dataset.csv', index=False)
    return None


class dataset(Dataset):
    """Brain-age fine-tuning dataset"""

    def __init__(self, csv_file, transform = None):
        self.file_frame = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.file_frame)

    def __getitem__(self, idx):
        stack_name = self.file_frame.iloc[idx]['processed_file_name']
        tensor = self.transform(stack_name)  
        tensor = (tensor - tensor.mean())/tensor.std()
        tensor = torch.clamp(tensor,-1, 5)
        age = self.file_frame.iloc[idx]['Age']   
        return tensor, age
    
def get_train_valid_loader(csv_file,
                           batch_size=4,
                           random_seed=10,
                           aug='none'):
    if aug == 'none':
        train_transforms = Compose([LoadNifti(image_only=True), ToTensor()])
    elif aug == 'flip':
        train_transforms = Compose([LoadNifti(image_only=True), 
                                RandFlip(prob=0.5, spatial_axis=0),
                                ToTensor()])
        
    valid_transforms = Compose([LoadNifti(image_only=True), ToTensor()])
    test_transforms = Compose([LoadNifti(image_only=True), ToTensor()])
   
    train_dataset = dataset(csv_file, transform=train_transforms)   
    valid_dataset = dataset(csv_file, transform=valid_transforms)
    test_dataset = dataset(csv_file, transform=test_transforms)
                         
    df = pd.read_csv(csv_file)
    IDs = df['ID'].unique().tolist()
    
    train_ids, test_ids = train_test_split(IDs, test_size=0.2, random_state=random_seed)
    train_ids, valid_ids = train_test_split(train_ids, test_size=0.2, random_state=random_seed)
    
    train_idx = df[df['ID'].isin(train_ids)].index.tolist()
    valid_idx = df[df['ID'].isin(valid_ids)].index.tolist()
    test_idx = df[df['ID'].isin(test_ids)].index.tolist()
           
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    #Creating intsances of training and validation dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    print('Number of training scans: {}, valid scans: {}, test scans: {}'.format(len(train_idx), len(valid_idx), len(test_idx)))
    return train_loader, valid_loader, test_loader
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--skull_strip', dest='skull_strip', action='store_true')
    parser.set_defaults(skull_strip=False)
    parser.add_argument('--already_processed', dest='already_processed', action='store_true')
    parser.set_defaults(already_processed=False)
    parser.add_argument('--aug', type=str, default='flip')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--sequence', type=str, default='t2')    
        
        
    args = parser.parse_args()
    save_dir = './{}/'.format(args.project_name)
    if args.already_processed:
        assert os.path.exists(save_dir + 'fine_tuning_dataset.csv'), ''' Couldn't find csv file for processed nii files at {}/fine_tuning_dataset.csv'''.format(save_dir)
        train_loader, valid_loader, test_loader = get_train_valid_loader(save_dir + 'fine_tuning_dataset.csv',
                           batch_size=args.batch_size,
                           random_seed=args.seed,
                           aug=args.aug)
    else:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            nii_dir = save_dir + 'processed_nii'
            os.mkdir(nii_dir)
        else:
            raise ValueError('Project name {} already used'.format(args.project_name)) 
            
        _ = process(args.csv_file, args.project_name, args.sequence, save_dir, args.skull_strip)                     
                                            
        train_loader, valid_loader, test_loader = get_train_valid_loader(save_dir + 'fine_tuning_dataset.csv',
                           batch_size=args.batch_size,
                           random_seed=args.seed,
                           aug=args.aug)
        
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
                        
    model_save_path = save_dir + datetime.datetime.now().strftime('{}_%d-%m-%y-%H_%M.pt'.format(args.sequence))

    if args.sequence == 't2':
        if args.skull_strip:
            state_dict = convert_state_dict('./Models/T2/Skull_stripped/seed_42.pt')
            net = DenseNet(3,1,1)
            net.load_state_dict(state_dict)
            net = net.to(device)
        else:
            state_dict = convert_state_dict('./Models/T2/Raw/seed_42.pt')
            net = DenseNet(3,1,1)
            net.load_state_dict(state_dict)
            net = net.to(device)
    elif args.sequence == 't1':
        if args.skull_strip:
            state_dict = convert_state_dict('./Models/T1/Skull_stripped/seed_60.pt')
            net = DenseNet(3,1,1)
            net.load_state_dict(state_dict)
            net = net.to(device)
        else:
            raise ValueError('Raw T1 model not currently handled. Please specify --skull_strip if skull-stripped and registered (MNI152) T1 model is desired')
    else:
        raise ValueError('{} MRI sequence not currently handled (must be one of t2 or t1; DWI and FLAIR coming soon!)'.format(args.sequence))    
        
    params =  net.parameters() 
    optimizer = optim.Adam(net.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion = nn.L1Loss()
    eval_criterion = nn.L1Loss(reduction='sum')
    out = train(net, optimizer, scheduler, train_loader, valid_loader, criterion, eval_criterion, model_save_path, epochs=60, patience=12)

    best_state_dict = torch.load(model_save_path)
    net.load_state_dict(best_state_dict)


    loss, corr, true_ages, pred_ages = evaluate(net, test_loader, eval_criterion)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.scatter(true_ages, pred_ages, alpha=0.3)
    ax.plot(true_ages, true_ages,linestyle= '--', color='black')
    #ax.set_ylim([min(true_ages), max(true_ages)])
    ax.set_aspect('equal')
    ax.set_xlabel('Chronological age')
    ax.set_ylabel('Predicted age')
    ax.set_title('MAE = {:.2f} years, p = {:.2f}\n'.format(loss, corr))
    fig.savefig(os.path.join(save_dir, 'fine_tune_scatter.png'))
plt.pause(0.1)
                      
    
    
    
