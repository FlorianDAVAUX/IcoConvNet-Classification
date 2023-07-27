import sys
sys.path.insert(0, './Utils')
sys.path.insert(0, './../Utils')

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics

import utils
from utils import ReadSurf, PolyDataToTensors, CreateIcosahedron

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRendererWithFragments, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_scene

import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import seaborn

import cv2

from IcoConcOperator import IcosahedronConv1d, IcosahedronConv2d, IcosahedronLinear
from layers import GaussianNoise, MaxPoolImages, AvgPoolImages, SelfAttention, Identity, TimeDistributed



class IcoConvNet(pl.LightningModule):
    def __init__(self,Layer,pretrained,nbr_features,nbr_demographic,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size,weights,radius=1,lr=1e-4,name=''):
        print('Inside init function')
        super().__init__()

        self.save_hyperparameters()

        self.Layer = Layer
        self.pretrained = pretrained
        self.nbr_features = nbr_features
        self.nbr_demographic = nbr_demographic
        self.dropout_lvl = dropout_lvl
        self.image_size = image_size
        self.noise_lvl = noise_lvl
        self.batch_size = batch_size
        self.weights = weights
        self.radius = radius
        self.name = name

        self.y_pred = []
        self.y_true = []

        ico_sphere = utils.CreateIcosahedron(self.radius, ico_lvl)
        ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = np.array(self.ico_sphere_edges)
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])):
               R_current = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)
        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)


        self.drop = nn.Dropout(p=self.dropout_lvl)
        self.noise = GaussianNoise(mean=0.0, std=noise_lvl)

        #####Left path

        efficient_netL = models.resnet18()
        efficient_netL.conv1 = nn.Conv2d(self.nbr_features, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        efficient_netL.fc = Identity()
        self.TimeDistributedL = TimeDistributed(efficient_netL)
        output_size = self.TimeDistributedL.module.inplanes

        if self.Layer == 'Att':
            out_size = 256
            self.WVL = nn.Linear(output_size, out_size) #256
            self.AttentionL = SelfAttention(output_size, 128) #128
        elif self.Layer == 'IcoConv2D':
            out_size = 256
            conv2dL = nn.Conv2d(output_size, out_size, kernel_size=(3,3),stride=2,padding=0)
            self.IcosahedronConv2dL = IcosahedronConv2d(conv2dL,self.ico_sphere_verts,self.ico_sphere_edges)
            self.poolingL = AvgPoolImages(nbr_images=self.nbr_cam)
        elif self.Layer == 'IcoConv1D':
            out_size = 256
            conv1dL = nn.Conv1d(output_size, out_size,7)
            self.IcosahedronConv2dL = IcosahedronConv1d(conv1dL,self.ico_sphere_verts,self.ico_sphere_edges)
            self.poolingL = AvgPoolImages(nbr_images=self.nbr_cam)
        elif self.Layer == 'IcoLinear':
            out_size = 256
            linear_layerL = nn.Linear(output_size*7, out_size)
            self.IcosahedronConv2dL = IcosahedronLinear(linear_layerL,self.ico_sphere_verts,self.ico_sphere_edges)
            self.poolingL = AvgPoolImages(nbr_images=self.nbr_cam)

        #####Right path

        efficient_netR = models.resnet18()
        efficient_netR.conv1 = nn.Conv2d(self.nbr_features, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        efficient_netR.fc = Identity()
        self.TimeDistributedR = TimeDistributed(efficient_netR)

        if self.Layer == 'Att':
            out_size = 256
            self.WVR = nn.Linear(output_size, out_size) 
            self.AttentionR = SelfAttention(output_size, 128) 
        elif self.Layer == 'IcoConv2D':
            out_size = 256
            conv2dR = nn.Conv2d(output_size, out_size, kernel_size=(3,3),stride=2,padding=0)
            self.IcosahedronConv2dR = IcosahedronConv2d(conv2dR,self.ico_sphere_verts,self.ico_sphere_edges)
            self.poolingR = AvgPoolImages(nbr_images=self.nbr_cam)
        elif self.Layer == 'IcoConv1D':
            out_size = 256
            conv1dL = nn.Conv1d(output_size, out_size, 7)
            self.IcosahedronConv2dR = IcosahedronConv1d(conv1dL,self.ico_sphere_verts,self.ico_sphere_edges)
            self.poolingR = AvgPoolImages(nbr_images=self.nbr_cam)
        elif self.Layer == 'IcoLinear': 
            out_size = 256   
            linear_layerR = nn.Linear(output_size*7, out_size)
            self.IcosahedronConv2dR = IcosahedronLinear(linear_layerR,self.ico_sphere_verts,self.ico_sphere_edges)
            self.poolingR = AvgPoolImages(nbr_images=self.nbr_cam)

        #Demographics
        self.normalize = nn.BatchNorm1d(self.nbr_demographic)


        #Final layer
        self.Classification = nn.Linear(2*out_size+self.nbr_demographic, 2)

        #Loss
        self.loss_train = nn.CrossEntropyLoss(weight=self.weights[0])
        self.loss_val = nn.CrossEntropyLoss(weight=self.weights[1])
        self.loss_test = nn.CrossEntropyLoss(weight=self.weights[2])

        #Accuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=2,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=2,average='macro')

        
        # Initialize a perspective camera.
        self.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0,
            faces_per_pixel=1,
            max_faces_per_bin=100000
        )

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            )
        self.phong_renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer




    def forward(self, x):

        VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic = x

        ###To Device
        VL = VL.to(self.device,non_blocking=True)
        FL = FL.to(self.device,non_blocking=True)
        VFL = VFL.to(self.device,non_blocking=True)
        FFL = FFL.to(self.device,non_blocking=True)
        VR = VR.to(self.device,non_blocking=True)
        FR = FR.to(self.device,non_blocking=True)
        VFR = VFR.to(self.device,non_blocking=True)
        FFR = FFR.to(self.device,non_blocking=True)
        demographic = demographic.to(self.device,non_blocking=True)

        ###Resnet18+Ico+Concatenation
        x = self.get_features(VL,FL,VFL,FFL,VR,FR,VFR,FFR,demographic)

        ###Last classification layer
        x = self.drop(x)
        x = self.Classification(x)

        return x

    def get_features(self,VL,FL,VFL,FFL,VR,FR,VFR,FFR,demographic):
        #########Left path
        xL, PF = self.render(VL,FL,VFL,FFL)   

        B,NV,C,L,W = xL.size()
        xL = self.TimeDistributedL(xL)
 
        if self.Is_it_Icolayer(self.Layer):
            xL = self.IcosahedronConv2dL(xL)
            xL = self.poolingL(xL)
        else:
            valuesL = self.WVL(xL)
            xL, score = self.AttentionL(xL,valuesL)            

        ###########Right path
        xR, PF = self.render(VR,FR,VFR,FFR)

        xR = self.TimeDistributedR(xR)


        if self.Is_it_Icolayer(self.Layer):
            xR = self.IcosahedronConv2dR(xR)
            xR = self.poolingR(xR)
        else:
            valuesR = self.WVR(xR)
            xR,score = self.AttentionR(xR,valuesR)   

        #Concatenation   

        demographic = self.normalize(demographic)

        l_left_right = [xL,xR,demographic]

        x = torch.cat(l_left_right,dim=1)

        return x

    def render(self,V,F,VF,FF):
        textures = TexturesVertex(verts_features=VF)
        meshes = Meshes(
            verts=V,
            faces=F,
            textures=textures
        )


        PF = []
        for i in range(self.nbr_cam):
            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))

        PF = torch.cat(PF, dim=1)
        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,:,index],PF)*(PF >= 0)) # take each feature for each pictures
        x = torch.cat(l_features,dim=2)

        return x, PF

    def training_step(self, train_batch, batch_idx):

        VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic, Y = train_batch

        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic))

        Y = Y.squeeze(dim=1)  

        loss = self.loss_train(x,Y)

        
        self.log('train_loss', loss) 
        predictions = torch.argmax(x, dim=1, keepdim=True)
        
    
        self.train_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy)           

        return loss

    def validation_step(self,val_batch,batch_idx):

        VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic, Y = val_batch

        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic))

        Y = Y.squeeze(dim=1) 

        loss = self.loss_val(x,Y)

        self.log('val_loss', loss)
        predictions = torch.argmax(x, dim=1, keepdim=True)

        val_acc = self.val_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("val_acc", val_acc)   

        return val_acc



    def test_step(self,test_batch,batch_idx):

        VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic, Y = test_batch


        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic))

        Y = Y.squeeze(dim=1)     

        loss = self.loss_test(x,Y)

        self.log('test_loss', loss, batch_size=self.batch_size)
        predictions = torch.argmax(x, dim=1, keepdim=True)

        output = [predictions,Y]

        return output

    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['No ASD','ASD']

        self.y_pred =y_pred
        self.y_true =y_true

        #Classification report
        print(self.y_pred)
        print(self.y_true)
        print(classification_report(self.y_true, self.y_pred, target_names=target_names))



    def GetView(self,meshes,index):

        phong_renderer = self.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)

        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face
        pix_to_face = pix_to_face.permute(0,3,1,2)
        return pix_to_face


    def get_y_for_report_classification(self):
        #This function could be called only after test step was done
        return (self.y_pred,self.y_true)
    
    def Is_it_Icolayer(self,layer):
        return (layer[:3] == 'Ico')