import numpy as np
import torch
import pytorch3d

import utils
from utils import GetUnitSurf, RandomRotation

class RotationTransform:
    def __call__(self, verts, rotation_matrix):
        verts = torch.transpose(torch.mm(rotation_matrix,torch.transpose(verts,0,1)),0,1)
        return verts

class RandomRotationTransform:
    def __call__(self, verts):
        rotation_matrix = pytorch3d.transforms.random_rotation()
        rotation_transform = RotationTransform()
        verts = rotation_transform(verts,rotation_matrix)
        return verts

class ApplyRotationTransform:
    def __init__(self):            
        self.rotation_matrix = pytorch3d.transforms.random_rotation()

    def __call__(self, verts):
        rotation_transform = RotationTransform()
        verts = rotation_transform(verts,self.rotation_matrix)
        return verts
    
    def change_rotation(self):
        self.rotation_matrix = pytorch3d.transforms.random_rotation()

class GaussianNoisePointTransform:
    def __init__(self, mean=0.0, std = 0.1):            
        self.mean = mean
        self.std = std

    def __call__(self, verts):
        noise = np.random.normal(loc=self.mean, scale=self.std, size=verts.shape)
        verts = verts + noise
        verts = verts.type(torch.float32)
        return verts

class NormalizePointTransform:
    def __call__(self, verts, scale_factor=1.0):
        bounds_max_v = [0.0] * 3
        v = torch.Tensor(verts)
        bounds = torch.tensor([torch.max(v[:,0]),torch.max(v[:,1]),torch.max(v[:,2])])
        bounds_max_v[0] = bounds[0]
        bounds_max_v[1] = bounds[1]
        bounds_max_v[2] = bounds[2]
        scale_factor = torch.tensor(bounds_max_v)

        verts = torch.multiply(v, 1/scale_factor)
        verts = verts.type(torch.float32)

        return verts


class CenterTransform:
    def __call__(self, verts,mean_arr = None):
        #calculate bounding box
        mean_v = [0.0] * 3

        v = torch.Tensor(verts)

        bounds = torch.tensor([torch.min(v[:,0]),torch.max(v[:,0]),torch.min(v[:,1]),torch.max(v[:,1]),torch.min(v[:,2]),torch.max(v[:,2])])

        mean_v[0] = (bounds[0] + bounds[1])/2.0
        mean_v[1] = (bounds[2] + bounds[3])/2.0
        mean_v[2] = (bounds[4] + bounds[5])/2.0
        
        #centering points of the shape
        mean_arr = torch.tensor(mean_v)

        verts = verts - mean_arr
        return verts 
       