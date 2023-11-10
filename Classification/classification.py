import sys
sys.path.insert(0, './../Utils')
import argparse

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

def main(args,arg_groups):

    list_path_ico = [args.path_ico_left,args.path_ico_right]

    ###Demographics
    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight'] #MLR

    ###Transformation
    list_train_transform = [] 
    list_train_transform.append(CenterTransform())
    list_train_transform.append(NormalizePointTransform())
    list_train_transform.append(RandomRotationTransform())        
    list_train_transform.append(GaussianNoisePointTransform(args.mean,args.std)) # Don't use this transformation if your object isn't a sphere
    list_train_transform.append(NormalizePointTransform()) # Don't use this transformation if your object isn't a sphere
    train_transform = monai.transforms.Compose(list_train_transform)

    list_val_and_test_transform = []    
    list_val_and_test_transform.append(CenterTransform())
    list_val_and_test_transform.append(NormalizePointTransform())
    val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

    ### Get number of images
    list_nb_verts_ico = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
    nb_images = list_nb_verts_ico[args.ico_lvl-1]
    
    ### Creation of Dataset
    brain_data = BrainIBISDataModule(args.batch_size,list_demographic,args.path_data,args.data_train,args.data_val,args.data_test,list_path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=args.num_workers)#MLR
    nbr_features = brain_data.get_features()
    weights = brain_data.get_weigths()
    nbr_demographic = brain_data.get_nbr_demographic()

    if args.ico_lvl == 1:
        radius = 1.76 
    elif args.ico_lvl == 2:
        radius = 1
    #Creation of our model
    model = IcoConvNet(args.layer,args.pretrained,nbr_features,nbr_demographic,args.dropout_lvl,args.image_size,args.noise_lvl,args.ico_lvl,args.batch_size,weights,radius=radius,lr=args.lr,name=args.name)#MLR

    #Creation of Checkpoint (if we want to save best models)
    checkpoint_callback_loss = ModelCheckpoint(dirpath='../Checkpoint/'+args.name,filename='{epoch}-{val_loss:.2f}',save_top_k=10,monitor='val_loss',)

    #Logger (Useful if we use Tensorboard)
    logger = TensorBoardLogger(save_dir="test_tensorboard", name="my_model")

    #Early Stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_early_stopping, patience=args.patience_early_stopping, verbose=True, mode="min")

    #Image Logger (Useful if we use Tensorboard)
    image_logger = ImageLogger(num_features = nbr_features,num_images = nb_images,mean = 0,std=args.noise_lvl)

    ###Trainer
    trainer = Trainer(log_every_n_steps=10,reload_dataloaders_every_n_epochs=True,logger=logger,max_epochs=args.num_epochs,callbacks=[early_stop_callback,checkpoint_callback_loss,image_logger],accelerator="gpu") #,accelerator="gpu"
    trainer.fit(model,datamodule=brain_data)
    trainer.test(model, datamodule=brain_data)

    print('Number of features : ',nbr_features)




def cml():
    #Command line arguments
    parser = argparse.ArgumentParser(description='IcoConv : Brain cortical surface analysis')
    
    ##Hyperparameters
    hyperparameters_group = parser.add_argument_group('Hyperparameters')
    hyperparameters_group.add_argument('--batch_size', type=int, default=2, help='Input batch size for training (default: 2)')
    hyperparameters_group.add_argument('--num_workers', type=int, default=12, help='Number of workers (default: 12)')
    hyperparameters_group.add_argument('--image_size', type=int, default=224, help='Image size (default: 224)')
    hyperparameters_group.add_argument('--noise_lvl', type=float, default=0.01, help='Noise level (default: 0.01)')
    hyperparameters_group.add_argument('--dropout_lvl', type=float, default=0.2, help='Dropout level (default: 0.2)')
    hyperparameters_group.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs (default: 1000)')
    hyperparameters_group.add_argument('--ico_lvl', type=int, default=2, help='Ico level, minimum level is 1 (default: 2)')
    hyperparameters_group.add_argument('--pretrained', type=bool, default=False, help='Pretrained (default: False)')
    hyperparameters_group.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')

    ##Gaussian Filter
    gaussian_group = parser.add_argument_group('Gaussian filter')
    gaussian_group.add_argument('--mean', type=float, default=0, help='Mean (default: 0)')
    gaussian_group.add_argument('--std', type=float, default=0.005, help='Standard deviation (default: 0.005)')

    ##Early Stopping
    early_stopping_group = parser.add_argument_group('Early stopping')
    early_stopping_group.add_argument('--min_delta_early_stopping', type=float, default=0.00, help='Minimum delta (default: 0.00)')
    early_stopping_group.add_argument('--patience_early_stopping', type=int, default=100, help='Patience (default: 100)')

    ##Paths
    paths_group = parser.add_argument_group('Paths to data')
    paths_group.add_argument('--path_data', help='Path to data',type=str, required=True)
    paths_group.add_argument('--data_train', help='Path to train data',type=str, required=True)
    paths_group.add_argument('--data_val', help='Path to validation data',type=str, required=True)
    paths_group.add_argument('--data_test', help='Path to test data',type=str, required=True)
    paths_group.add_argument('--path_ico_left', type=str, default='../3DObject/sphere_f327680_v163842.vtk', help='Path to ico left (default: ../3DObject/sphere_f327680_v163842.vtk)')
    paths_group.add_argument('--path_ico_right', type=str, default='../3DObject/sphere_f327680_v163842.vtk', help='Path to ico right (default: ../3DObject/sphere_f327680_v163842.vtk)')

    ##Name and layer
    name_group = parser.add_argument_group('Name and layer')
    name_group.add_argument('--layer', type=str, default='IcoConv2D', help="Layer, choose between 'Att','IcoConv2D','IcoConv1D','IcoLinear' (default: IcoConv2D)")
    name_group.add_argument('--name', type=str, default='Experiment0', help='Name of your experiment (default: Experiment0)')

    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        arg_groups[group.title] = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}

    main(args,arg_groups)



if __name__ == '__main__':
    cml()