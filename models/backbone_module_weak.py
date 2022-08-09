# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule  

class Pointnet2Backbone_weak(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. #SSG方式
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,  
                radius=0.2,
                nsample=64,  
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,  
                normalize_xyz=True  
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])
        #*****************************************
        self.fp3 = PointnetFPModule(mlp=[256+128,256,128])
        self.fp4 = PointnetFPModule(mlp=[128+input_feature_dim,128,128,128])
        
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 18, kernel_size=1),
        )
        


    def _break_up_pc(self, pc):
        """
        将pc分成坐标和特征
        """
        xyz = pc[..., 0:3].contiguous()  
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]  
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        #****************************
        xyz_raw = xyz
        feature_raw = features
        #****************************

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds_weak'] = fps_inds
        end_points['sa1_xyz_weak'] = xyz
        end_points['sa1_features_weak'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds_weak'] = fps_inds
        end_points['sa2_xyz_weak'] = xyz
        end_points['sa2_features_weak'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz_weak'] = xyz
        end_points['sa3_features_weak'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz_weak'] = xyz
        end_points['sa4_features_weak'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz_weak'], end_points['sa4_xyz_weak'], end_points['sa3_features_weak'], end_points['sa4_features_weak'])
        features = self.fp2(end_points['sa2_xyz_weak'], end_points['sa3_xyz_weak'], end_points['sa2_features_weak'], features)
        end_points['fp2_features_weak'] = features
        end_points['fp2_xyz_weak'] = end_points['sa2_xyz_weak']
        num_seed = end_points['fp2_xyz_weak'].shape[1] 
        end_points['fp2_inds_weak'] = end_points['sa1_inds_weak'][:,0:num_seed] # indices among the entire input point clouds
        
        #****************************
        features = self.fp3(end_points['sa1_xyz_weak'], end_points['sa2_xyz_weak'], end_points['sa1_features_weak'], features)
        features = self.fp4(xyz_raw, end_points['sa1_xyz_weak'], feature_raw, features)

        semantic_pre = self.fc_layer(features)  # batch x channel x kpoints -> batch x num_class x kpoints
        semantic_pre = semantic_pre.transpose(2, 1)
        end_points['semantic_score_map'] = semantic_pre #(B,N,18)
        end_points['all_semantic_point'] = xyz_raw
        #****************************
        return end_points


if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
