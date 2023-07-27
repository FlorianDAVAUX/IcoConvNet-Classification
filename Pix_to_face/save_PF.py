import sys
sys.path.insert(0, './../Utils')
sys.path.insert(0, '/work/ugor/source/brain_classification/SpectFormer')
import numpy as np 
import torch

import monai
import utils

from data import BrainIBISDataModule
from transformation import RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform

import monai

# datastructures
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRendererWithFragments, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_scene






###Getview
def GetView(renderer,R,T,meshes,index,device):

    renderer = renderer.to(device)
    R = R[index][None].to(device)
    T = T[index][None].to(device)

    _, fragments = renderer(meshes.clone(),R=R,T=T)
    pix_to_face = fragments.pix_to_face
    pix_to_face = pix_to_face.permute(0,3,1,2)
    return pix_to_face







###The level of icosahedron
ico_lvl = 1 
list_nb_verts_ico = [12,42,162, 642, 2562, 10242, 40962, 163842]
nb_images = list_nb_verts_ico[ico_lvl-1]

###Init Dataset
batch_size = 5
num_workers = 12 
image_size = 224
if ico_lvl == 1:
    radius = 1.76 
elif ico_lvl == 2:
    radius = 1
lr = 1e-4

###Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###Paths
path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"

data_train = "../Data/V06-12_train.csv"
data_val = "../Data/V06-12_val.csv"
data_test = "../Data/V06-12_test.csv"

path_ico_left = '../Icosahedron_template/sphere_f327680_v163842.vtk'
path_ico_right = '../sphere_f327680_v163842.vtk'
list_path_ico = [path_ico_left,path_ico_right]

###Transformation
list_train_transform = [] 
list_train_transform.append(CenterTransform())
list_train_transform.append(NormalizePointTransform())

train_transform = monai.transforms.Compose(list_train_transform)

list_val_and_test_transform = []    
list_val_and_test_transform.append(CenterTransform())
list_val_and_test_transform.append(NormalizePointTransform())

val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

###Resampling
resampling = 'no_resampling' #'no_resampling','min','max'
#Choose between these 3 choices to balence your data. Per default : no_resampling.


list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight']#MLR

brain_data = BrainIBISDataModule(batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,resampling=resampling,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)#MLR


ico_sphere = utils.CreateIcosahedron(radius, ico_lvl)
ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
ico_sphere_verts = ico_sphere_verts
ico_sphere_edges = np.array(ico_sphere_edges)
R=[]
T=[]
for coords_cam in ico_sphere_verts.tolist():
    camera_position = torch.FloatTensor([coords_cam])
    R_current = look_at_rotation(camera_position)
    # check if camera coords vector and up vector for R are collinear
    if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])):
        R_current = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)
    T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

    R.append(R_current)
    T.append(T_current)
R=torch.cat(R)
T=torch.cat(T)
nbr_cam = len(R)

# Initialize a perspective camera.
cameras = FoVPerspectiveCameras()

# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=image_size,
    blur_radius=0,
    faces_per_pixel=1,
    max_faces_per_bin=100000
)

lights = AmbientLights()
rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
phong_renderer = MeshRendererWithFragments(
    rasterizer=rasterizer,
    shader=HardPhongShader(cameras=cameras, lights=lights)
        )







VL, FL, VFL, FFL,VR, FR, VFR, FFR, demographic, Y = brain_data.train_dataset.__getitem__(0)

VL = VL.to(device,non_blocking=True).unsqueeze(dim=0)
FL = FL.to(device,non_blocking=True).unsqueeze(dim=0)
VFL = VFL.to(device,non_blocking=True).unsqueeze(dim=0)
FFL = FFL.to(device,non_blocking=True).unsqueeze(dim=0)
VR = VR.to(device,non_blocking=True).unsqueeze(dim=0)
FR = FR.to(device,non_blocking=True).unsqueeze(dim=0)
VFR = VFR.to(device,non_blocking=True).unsqueeze(dim=0)
FFR = FFR.to(device,non_blocking=True).unsqueeze(dim=0)



textures = TexturesVertex(verts_features=VFL)
meshes = Meshes(
    verts=VL,
    faces=FL,
    textures=textures
)




PF = []
for i in range(nbr_cam):
    pix_to_face = GetView(phong_renderer,R,T,meshes,i,device)
    PF.append(pix_to_face.unsqueeze(dim=1))
PF = torch.cat(PF, dim=1)
PF = PF.squeeze(dim=2).squeeze(dim=0)

name_save = 'PF'+str(nb_images)+'.pt'
torch.save(PF,name_save)