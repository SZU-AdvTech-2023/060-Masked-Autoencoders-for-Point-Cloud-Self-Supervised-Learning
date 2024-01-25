import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not BASE_DIR in sys.path:
    sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from AMS import AngleLinear
import numpy as np

# Relation-Shape CNN: Single-Scale Neighborhood
class RSCNN_SSN_SIAMESE(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=0, relation_prior=1, use_xyz=True, local=True):
        super().__init__()

        self.need_local = local
        self.SA_modules = nn.ModuleList()
        
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=2048, #1024
                radii=[0.24], #0.23
                nsamples=[32], #48 32
                mlps=[[input_channels, 64]], #128
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024, #512
                radii=[0.28], # 0.32 0.28
                nsamples=[48], #64 48
                mlps=[[64, 128]], # 256
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512, #256
                radii=[0.32], #0.32
                nsamples=[64], #64
                mlps=[[128, 256]], #512
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        """ # No.4 SA layer"""
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256, #128
                radii=[0.32],
                nsamples=[64],
                mlps=[[256, 512]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample = 256, 
                mlp=[512, 1024], 
                use_xyz=use_xyz
            )
        )

        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.4),
            # pt_utils.FC(512, 512, activation=nn.ReLU(inplace=True), bn=True),
            # nn.Dropout(p=0.5),
            #pt_utils.FC(512, num_classes, activation=None)
        )# I modified in 4/8, 2020
        self.finalLinear = pt_utils.FC(512, num_classes, activation=None)


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud):
        for i, module in enumerate(self.SA_modules):
            #if i == 1 and self.need_local==True:
            #    _, features_local_A = module(xyz_A, features_A)
            #    _, features_local_B = module(xyz_B, features_B)
            xyz_A, features_A = module(xyz_A, features_A)

        mediate_features_A = self.FC_layer(features_A.squeeze(-1))

        pred_A = self.finalLinear(mediate_features_A)
        
        return mediate_features_A, mediate_features_B
        

        

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    device = torch.device("cuda:0")
    sample_data_a = torch.randn(4, 4096, 3).to(device)
    sample_data_b = torch.randn(4, 4096, 3).to(device)  
    model = RSCNN_SSN_SIAMESE(1000).to(device)  
    from torchsummaryX import summary
    summary(model, (sample_data_a, sample_data_b))
    # mf_a, pred_a, mf_b, pred_b, local_a, local_b = model((sample_data_a, sample_data_b))
    # print(mf_a.shape, pred_a)
    