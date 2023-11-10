import numpy as np
import random
import torch
import os 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl 
from torch.nn.functional import normalize

import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight

import utils
from utils import ReadSurf, PolyDataToTensors

import pandas as pd


class BrainIBISDataset(Dataset):
    def __init__(self,df,list_demographic,path_data,list_path_ico,transform = None,version=None,column_subject_ID='Subject_ID',column_age='Age',name_class = 'ASD_administered'):
        self.df = df
        self.list_demographic = list_demographic
        self.path_data = path_data
        self.list_path_ico = list_path_ico
        self.transform = transform
        self.version = version
        self.column_subject_ID = column_subject_ID
        self.column_age = column_age
        self.name_class = name_class

    def __len__(self):
        return(len(self.df)) 

    def __getitem__(self,idx):
        #Get item for each hemisphere (left and right)
        vertsL, facesL, vertex_featuresL, face_featuresL,demographic,Y = self.getitem_per_hemisphere('left', idx)
        vertsR, facesR, vertex_featuresR, face_featuresR,demographic,Y = self.getitem_per_hemisphere('right', idx)
        return  vertsL, facesL, vertex_featuresL, face_featuresL, vertsR, facesR, vertex_featuresR, face_featuresR,demographic, Y 
    
    def data_to_tensor(self,path):
        data = open(path,"r").read().splitlines()
        data = torch.tensor([float(ele) for ele in data])
        return data

    def getitem_per_hemisphere(self,hemisphere,idx):
        #Load Data
        row = self.df.loc[idx]
        path_left_eacsf = row['PathLeftEACSF']
        path_right_eacsf = row['PathRightEACSF']
        path_left_sa = row['PathLeftSa']
        path_right_sa = row['PathRightSa']
        path_left_thickness = row['PathLeftThickness']
        path_right_thickness = row['PathRightThickness']

        l_features = []

        if hemisphere == 'left':
            l_features.append(self.data_to_tensor(path_left_eacsf).unsqueeze(dim=1))
            l_features.append(self.data_to_tensor(path_left_sa).unsqueeze(dim=1))
            l_features.append(self.data_to_tensor(path_left_thickness).unsqueeze(dim=1))
        else:
            l_features.append(self.data_to_tensor(path_right_eacsf).unsqueeze(dim=1))
            l_features.append(self.data_to_tensor(path_right_sa).unsqueeze(dim=1))
            l_features.append(self.data_to_tensor(path_right_thickness).unsqueeze(dim=1))

        vertex_features = torch.cat(l_features,dim=1)

        #Demographics
        demographic_values = [float(row[name]) for name in self.list_demographic]
        demographic = torch.tensor(demographic_values)

        #Y
        Y = torch.tensor([int(row[self.name_class])])

        #Load  Icosahedron

        if hemisphere == 'left':
            reader = utils.ReadSurf(self.list_path_ico[0])
        else:
            reader = utils.ReadSurf(self.list_path_ico[1])
        verts, faces, edges = utils.PolyDataToTensors(reader)

        nb_faces = len(faces)

        #Transformations
        if self.transform:        
            verts = self.transform(verts)

        #Face Features
        faces_pid0 = faces[:,0:1]         
    
        offset = torch.zeros((nb_faces,vertex_features.shape[1]), dtype=int) + torch.Tensor([i for i in range(vertex_features.shape[1])]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])      
        
        face_features = torch.take(vertex_features,faces_pid0_offset)

        return verts, faces,vertex_features,face_features,demographic, Y


class BrainIBISDataModule(pl.LightningDataModule):
    def __init__(self,batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,train_transform=None,val_and_test_transform=None, num_workers=6, pin_memory=False, persistent_workers=False,name_class='ASD_administered'):
        super().__init__()
        self.batch_size = batch_size 
        self.list_demographic = list_demographic
        self.path_data = path_data
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.list_path_ico = list_path_ico
        self.train_transform = train_transform
        self.val_and_test_transform = val_and_test_transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.name_class = name_class

        ### weights computing
        self.weights = []
        self.df_train = pd.read_csv(self.data_train)
        self.df_val = pd.read_csv(self.data_val)
        self.df_test = pd.read_csv(self.data_test)
        self.weights = self.class_weights()

        self.setup()

    
    def class_weights(self):
        class_weights_train = self.compute_class_weights(self.data_train)
        class_weights_val = self.compute_class_weights(self.data_val)
        class_weights_test = self.compute_class_weights(self.data_test)
        return [class_weights_train, class_weights_val, class_weights_test]

    def compute_class_weights(self, data_file):
        df = pd.read_csv(data_file)
        y = np.array(df.loc[:, self.name_class])
        labels = np.unique(y)
        class_weights = torch.tensor(class_weight.compute_class_weight('balanced', classes=labels, y=y)).to(torch.float32)
        return class_weights

    def setup(self,stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = BrainIBISDataset(self.df_train,self.list_demographic,self.path_data,self.list_path_ico,self.train_transform)
        self.val_dataset = BrainIBISDataset(self.df_val,self.list_demographic,self.path_data,self.list_path_ico,self.val_and_test_transform)
        self.test_dataset = BrainIBISDataset(self.df_test,self.list_demographic,self.path_data,self.list_path_ico,self.val_and_test_transform)

        VL, FL, VFL, FFL,VR, FR, VFR, FFR, demographic, Y = self.train_dataset.__getitem__(0)
        self.nbr_features = VFL.shape[1]
        self.nbr_demographic = demographic.shape[0]

    def train_dataloader(self):    
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=True)

    def repeat_subject(self,df,final_size):
        n = len(df)
        q,r = final_size//n,final_size%n
        list_df = [df for i in range(q)]
        list_df.append(df[:r])
        new_df = pd.concat(list_df).reset_index().drop(['index'],axis=1)
        return new_df

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=True)        

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=True)

    def get_features(self):
        return self.nbr_features

    def get_weigths(self):
        return self.weights
    
    def get_nbr_demographic(self):
        return self.nbr_demographic

