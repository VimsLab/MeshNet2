import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PointDescriptor, NormalDescriptor
from models import ConvSurface
from models import MaxPoolFaceFeature, MeshBlock
from models import PsuedoMeshBlock

class MeshNet2(nn.Module):
    """ MeshNet++ Model"""
    def __init__(self, cfg, num_faces, num_cls, pool_rate):
        """
        Args:
            cfg: configuration file
            num_faces: number of mesh faces
            num_cls: number of classes in dataset
        """
        # Setup
        super(MeshNet2, self).__init__()
        self.pool_rate = pool_rate
        self.point_descriptor = PointDescriptor(num_kernel=cfg['num_kernel'])
        self.normal_descriptor = NormalDescriptor(num_kernel=cfg['num_kernel'])
        self.conv_surface_1 = ConvSurface(num_faces=num_faces, num_neighbor=3, cfg=cfg['ConvSurface'])
        self.conv_surface_2 = ConvSurface(num_faces=num_faces, num_neighbor=6, cfg=cfg['ConvSurface'])
        self.conv_surface_3 = ConvSurface(num_faces=num_faces, num_neighbor=12, cfg=cfg['ConvSurface'])

        blocks = cfg['MeshBlock']['blocks']
        in_channel = cfg['num_kernel'] * 2 + cfg['ConvSurface']['num_kernel'] * 3

        self.mesh_block_1 = MeshBlock(in_channel=in_channel,
                                      num_block=blocks[0],
                                      growth_factor=cfg['num_kernel'],
                                      num_neighbor=3)
        in_channel = in_channel + blocks[0] * cfg['num_kernel']
        self.max_pool_fea_1 = MaxPoolFaceFeature(in_channel=in_channel, num_neighbor=3)

        self.mesh_block_2 = PsuedoMeshBlock(in_channel=in_channel,
                                            num_block=blocks[1],
                                            growth_factor=cfg['num_kernel'],
                                            num_neighbor=6)
        in_channel = in_channel + blocks[1] * cfg['num_kernel']
        self.max_pool_fea_2 = MaxPoolFaceFeature(in_channel=in_channel, num_neighbor=6)

        self.mesh_block_3 = PsuedoMeshBlock(in_channel=in_channel,
                                            num_block=blocks[2],
                                            growth_factor=cfg['num_kernel'],
                                            num_neighbor=12)
        in_channel = in_channel + blocks[2] * cfg['num_kernel']

        self.classifier = nn.Sequential(
            nn.Linear(in_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_cls)
        )

        print('Spatial descriptor number of learnable kernels: {0}'.format(cfg['num_kernel']))
        print('Structural descriptor number of learnable kernels: {0}'.format(cfg['ConvSurface']['num_kernel']))
        print('Structural descriptor resampling mode: {0}'.format(cfg['ConvSurface']['rs_mode']))
        print('MeshNet2 pool rate: {0}'.format(self.pool_rate))

    def forward(self, verts, faces, centers, normals, ring_1, ring_2, ring_3):
        """
        Args:
            verts: padded mesh vertices
            [num_meshes, ?, 3]

            faces: faces in mesh/es
            [num_meshes, num_faces, 3]

            centers: face center of mesh/es
            [num_meshes, num_faces, 3]

            normals: face normals of mesh/es
            [num_meshes, num_faces, 3]

            ring_1: 1st Ring neighbors of faces
            [num_meshes, num_faces, 3]

            ring_2: 2nd Ring neighbors of faces
            [num_meshes, num_faces, 6]

            ring_3: 3rd Ring neighbors of faces
            [num_meshes, num_faces, 12]

        Returns:
            cls: predicted class of the input mesh/es
        """
        # Face center features
        points_fea = self.point_descriptor(centers=centers)

        # Face normal features
        normals_fea = self.normal_descriptor(normals=normals)

        # Surface features from 1-Ring neighborhood around a face
        surface_fea_1 = self.conv_surface_1(verts=verts,
                                            faces=faces,
                                            ring_n=ring_1,
                                            centers=centers)

        # Surface features from 2-Ring neighborhood around a face
        surface_fea_2 = self.conv_surface_2(verts=verts,
                                            faces=faces,
                                            ring_n=ring_2,
                                            centers=centers)

        # Surface features from 3-Ring neighborhood around a face
        surface_fea_3 = self.conv_surface_3(verts=verts,
                                            faces=faces,
                                            ring_n=ring_3,
                                            centers=centers)

        # Concatenate spatial and structural features
        fea_in = torch.cat([points_fea, surface_fea_1, surface_fea_2, surface_fea_3, normals_fea], 1)

        # Mesh block 1 features
        fea = self.mesh_block_1(fea=fea_in, ring_n=ring_1)

        # Max pool features
        fea = self.max_pool_fea_1(fea=fea, ring_n=ring_1)

        # Randomly select pooling indicies. Face indices not in pooling_idx will not be considered by
        # further layers.
        # Note: pooling_idx is same for all meshes and size of the orginal tensor does not change
        pool_idx = torch.randperm(ring_2.shape[1])[:ring_2.shape[1]//self.pool_rate]

        # Sort the index for correct tensor re-assignment in PsuedoMeshBlock
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 2 features
        fea = self.mesh_block_2(fea=fea, ring_n=ring_2, pool_idx=pool_idx)

        # Max pool features
        fea = self.max_pool_fea_2(fea=fea, ring_n=ring_2)

        # Randomly subset pooling indicies from initial pool_idx
        pool_idx_idx = torch.randperm(pool_idx.shape[0])[:pool_idx.shape[0]//self.pool_rate]
        pool_idx = pool_idx[pool_idx_idx]
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 3 features
        fea = self.mesh_block_3(fea=fea, ring_n=ring_3, pool_idx=pool_idx)

        # Only consider the pool_idx, global features
        fea = fea[:, :, pool_idx]

        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)
        cls = self.classifier(fea)
        return cls
