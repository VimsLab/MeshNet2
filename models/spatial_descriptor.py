""" MeshNet2 Spatial Descriptors """
import torch
import torch.nn as nn

class PointDescriptor(nn.Module):
    """
    r"Projects face centers into feature space by convolution
    """
    def __init__(self, num_kernel=64):
        """
        Args:
            num_kernel: dimension of feature space
        """
        super(PointDescriptor, self).__init__()
        self.num_kernel = num_kernel
        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, self.num_kernel, 1),
            nn.BatchNorm1d(self.num_kernel),
            nn.ReLU(),
        )

    def forward(self, centers):
        """
        Args:
            centers: x, y, and, z coordinates of face centers
            [num_meshes, num_faces, 3]

        Returns:
            mlp_centers: face centers in feature space
            [num_meshes, num_faces, num_kernel]
        """
        mlp_centers = self.spatial_mlp(centers)
        return mlp_centers
