import sys
sys.path.insert(0, './../Utils')

import numpy as np
import torch
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import monai
import nibabel as nib


from net import IcoConvNet
from data import BrainIBISDataModule
from logger import ImageLogger

from transformation import RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform


print("Import // done")

def main():

    ##############################################################################################Hyperparamters
    batch_size = 2
    num_workers = 12 
    image_size = 224
    noise_lvl = 0.01
    dropout_lvl = 0.2
    num_epochs = 1000
    ico_lvl = 2 #minimum level is 1
    pretrained = False #True,False
    if ico_lvl == 1:
        radius = 1.76 
    elif ico_lvl == 2:
        radius = 1
    lr = 1e-4
    print('lr : ',lr)

    #parameters for GaussianNoiseTransform
    mean = 0
    std = 0.005

    #parameters for EarlyStopping
    min_delta_early_stopping = 0.00
    patience_early_stopping = 100

    #Paths
    path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"

    data_train = "../Data/V06-12_train.csv"
    data_val = "../Data/V06-12_val.csv"
    data_test = "../Data/V06-12_test.csv"

    path_ico_left = '../3DObject/sphere_f327680_v163842.vtk'
    path_ico_right = '../3DObject/sphere_f327680_v163842.vtk'  
    list_path_ico = [path_ico_left,path_ico_right]

    ###Demographics
    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight']#MLR

    ###Transformation
    list_train_transform = [] 
    list_train_transform.append(CenterTransform())
    list_train_transform.append(NormalizePointTransform())
    list_train_transform.append(RandomRotationTransform())        
    list_train_transform.append(GaussianNoisePointTransform(mean,std)) # Don't use this transformation if your object isn't a sphere
    list_train_transform.append(NormalizePointTransform()) # Don't use this transformation if your object isn't a sphere

    train_transform = monai.transforms.Compose(list_train_transform)

    list_val_and_test_transform = []    
    list_val_and_test_transform.append(CenterTransform())
    list_val_and_test_transform.append(NormalizePointTransform())

    val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

    ###Layer
    Layer = 'IcoConv2D' #'Att','IcoConv2D','IcoConv1D','IcoLinear'
    #Choose between these 4 choices to choose what kind of model you want to use. 

    ###Name
    name = 'Experiment0'
    #name of your experiment

    

    ##############################################################################################
    

    ###Get number of images
    list_nb_verts_ico = [12,42,162, 642, 2562, 10242, 40962, 163842]
    nb_images = list_nb_verts_ico[ico_lvl-1]


    
    ###Creation of Dataset
    brain_data = BrainIBISDataModule(batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)#MLR
    nbr_features = brain_data.get_features()
    weights = brain_data.get_weigths()
    nbr_demographic = brain_data.get_nbr_demographic()


    #Creation of our model
    model = IcoConvNet(Layer,pretrained,nbr_features,nbr_demographic,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size,weights,radius=radius,lr=lr,name=name)#MLR

    #Creation of Checkpoint (if we want to save best models)
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath='../Checkpoint/'+name,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=10,
        monitor='val_loss',
    )

    #Logger (Useful if we use Tensorboard)
    logger = TensorBoardLogger(save_dir="test_tensorboard", name="my_model")

    #Early Stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode="min")

    print('nombre de features : ',nbr_features)

    #Image Logger (Useful if we use Tensorboard)
    image_logger = ImageLogger(num_features = nbr_features,num_images = nb_images,mean = 0,std=noise_lvl)



    ###Trainer
    trainer = Trainer(log_every_n_steps=10,reload_dataloaders_every_n_epochs=True,logger=logger,max_epochs=num_epochs,callbacks=[early_stop_callback,checkpoint_callback_loss,image_logger],accelerator="gpu") #,accelerator="gpu"

    trainer.fit(model,datamodule=brain_data)

    trainer.test(model, datamodule=brain_data)

if __name__ == '__main__':
    main()
