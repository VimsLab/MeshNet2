""" MeshNet2 Structural Descriptors """
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Softmax
import torch.nn.functional as F
import math
from models import check_ConvSurface

class NormalDescriptor(nn.Module):
    """
    r"Projects face normals into feature space by convolution
    """
    def __init__(self, num_kernel=64):
        """
        Args:
            num_kernel: dimension of feature space
        """
        super(NormalDescriptor, self).__init__()
        self.num_kernel = num_kernel
        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, self.num_kernel, 1),
            nn.BatchNorm1d(self.num_kernel),
            nn.ReLU(),
        )

    def forward(self, normals):
        """
        Args:
            normals: face normals
            [num_meshes, num_faces, 3]

        Returns:
            mlp_normals: face normals in feature space
            [num_meshes, num_faces, self.num_kernel]
        """
        mlp_normals = self.spatial_mlp(normals)
        return mlp_normals

class NeighborResampleUniform(nn.Module):
    r"""
    This class uniformly resamples the faces, edges, and corners in a n-Ring neighborhood.
    The number of neighbors (num_neighbor) in a n-Ring neighborhood is between 3-12.
    Example:
    If the resampling_frequency is three, then for the neighborhood illustrated below:
                          cf1
        ca2_ _ _ _ca3     /\     cc2_ _ _ _cc3
          \       /      /  \      \       /
           \  a  /  e1  / f  \  e2  \  c  /
            \   /      /_ _ __\      \   /
             ca1    cf2   e3   cf3    cc1
                    cb2_ _ _ _cb3
                      \       /
                       \  b  /
                        \   /
                         cb1
    Faces a, b, and c are resampled three times.
    Corners ca1, ca2, ca3, cb1, cb2, cb3, cc1, cc2, and cc3 are resampled three times.
    """
    def __init__(self, num_samples_per_neighbor=8):
        """
        Args:
            num_samples_per_neighbor: resampling frequency per neighbor
        """
        super(NeighborResampleUniform, self).__init__()
        self.resampling_frequency = num_samples_per_neighbor

    def forward(self, ring_n=None, neighbor_corners=None):
        """
        Args:
            ring_n: faces in a n-Ring neighborhood.
            [num_meshes, num_faces, num_neighbor]

            neighbor_corners: corners in a n-Ring neighborhood.
            [num_meshes, num_faces, num_neighbor, 3, 3]

        Returns:
            rs_ring_n: resampled n-Ring neighborhood faces
            [num_meshes, num_faces, num_neighbor*num_samples_per_neighbor]

            rs_neighbor_corners: resampled n-Ring neighborhood corners
            [num_meshes, num_faces, num_neighbor*num_samples_per_neighbor, 3, 3]
        """
        num_meshes = ring_n.shape[0]
        num_faces = ring_n.shape[1]
        num_neighbor = ring_n.shape[2]
        num_samples = num_neighbor * self.resampling_frequency

        # Duplicate/Stacking can be considered as uniform sampling
        rs_ring_n = torch.cat([ring_n]*self.resampling_frequency, 2)
        rs_neighbor_corners = torch.cat([neighbor_corners]*self.resampling_frequency, dim=2)

        # assert rs_ring_n.shape == (num_meshes, num_faces, num_samples)
        # assert rs_neighbor_corners.shape == (num_meshes, num_faces, num_samples, 3, 3)

        return rs_ring_n, rs_neighbor_corners

class NeighborResampleWeighted(nn.Module):
    r"""
    This class resamples faces and corners in a n-Ring neighborhood.
    Resampling frequency for faces is propotional to their perimeter.
    Once the faces are resampled, the corners for those faces are resampled.
    This ensures that corners for resampled neighbor faces are retained.

    Since perimeter of face b > c > a, then for the neighborhood illustrated below:
                          cf1
        ca2_ _ _ _ca3     / \     cc2_ _ _ _ _ _cc3
          \       /      /    \      \           /
           \  a  /  e1  / f     \  e2  \   c   /
            \   /      /_ _ _ _ _ \      \   /
             ca1    cf2   e3   cf3        cc1
                    cb2_ _ _ __ _cb3
                      \          /
                       \   b    /
                        \      /
                         \    /
                          \  /
                           cb1
    Number of times b resampled > c resampled > a resampled.
    """
    def __init__(self, num_samples=24):
        """
        Args:
            num_samples: count of neighbor samples after resampling
        """
        super(NeighborResampleWeighted, self).__init__()
        self.num_samples = num_samples

    def dist_neighbor_corners(self, neighbor_corners1=None, neighbor_corners2=None):
        """
        Helper function to compute euclidian distance between "two" corners of "same"
        neighbor face.
        Args:
            neighbor_corners1: x, y and z co-ordinates of corners in neighbor faces
            [num_meshes, num_faces, num_neighbor, 3]

            neighbor_corners2: x, y and z co-ordinates of corners in neighbor faces
            [num_meshes, num_faces, num_neighbor, 3]
        Returns:
            distance: euclidian distance between neighbor_corners1 and neighbor_corners2
            [num_meshes, num_faces, num_neighbor]
        """
        dist = torch.sqrt((neighbor_corners1[:, :, :, 0] - neighbor_corners2[:, :, :, 0])**2 +
                          (neighbor_corners1[:, :, :, 1] - neighbor_corners2[:, :, :, 1])**2 +
                          (neighbor_corners1[:, :, :, 2] - neighbor_corners2[:, :, :, 2])**2)
        return dist

    def forward(self, ring_n=None, neighbor_corners=None):
        """
        Args:
            ring_n: faces in a n-Ring neighborhood.
            [num_meshes, num_faces, num_neighbor]

            neighbor_corners: corners in a n-Ring neighborhood.
            [num_meshes, num_faces, num_neighbor, 3, 3]
        Returns:
            rs_ring_n: resampled n-Ring neighborhood faces
            [num_meshes, num_faces, num_samples]

            rs_neighbor_corners: resampled n-Ring neighborhood corners
            [num_meshes, num_faces, num_samples, 3, 3]
        """
        num_meshes = ring_n.shape[0]
        num_faces = ring_n.shape[1]
        num_neighbor = ring_n.shape[2]
        num_samples = self.num_samples - num_neighbor
        device = neighbor_corners.device
        #Initalize Placeholder
        rs_ring_n = torch.zeros((num_meshes, num_faces, num_samples), device=device)
        rs_neighbor_corners = torch.zeros((num_meshes, num_faces, num_samples, 3, 3), device=device)

        neighbor_corners1 = neighbor_corners[:, :, :, 0, :]
        neighbor_corners2 = neighbor_corners[:, :, :, 1, :]
        neighbor_corners3 = neighbor_corners[:, :, :, 2, :]
        #For each neighbor face compute the perimeter of triangular face
        perimeter = torch.stack([self.dist_neighbor_corners(neighbor_corners1, neighbor_corners2),
                                 self.dist_neighbor_corners(neighbor_corners2, neighbor_corners3),
                                 self.dist_neighbor_corners(neighbor_corners3, neighbor_corners1)
                                ], dim=3).sum(dim=3)

        # assert perimeter.shape == (num_meshes, num_faces, num_neighbor)

        #Expand tensor for advanced pytorch indexing
        num_faces_per_mesh = torch.arange(num_faces)[:, None, None]
        num_corners_per_face = torch.arange(3)[None, None, :, None]
        num_xyz_per_corner = torch.arange(3)[None, None, None, :]

        #For loop is required since PyTorch cannot perform multinomial sampling on 3D distribution
        for idx in range(num_meshes):
            # Index of neighbor face to be duplicate
            rs_idx = perimeter[idx].multinomial(num_samples, replacement=True)
            rs_ring_n[idx] = rs_idx
            #Expand tensor for advanced pytorch indexing
            rs_idx = rs_idx.unsqueeze(2)
            #Expand tensor for advanced pytorch indexing
            rs_idx = rs_idx.unsqueeze(3)
            rs_neighbor_corners[idx] = neighbor_corners[idx, :, :, :, :][num_faces_per_mesh.unsqueeze(3),
                                                                         rs_idx,
                                                                         num_corners_per_face,
                                                                         num_xyz_per_corner]

        #Garuntee that each neighbor is duplicated atleast once
        rs_ring_n = torch.cat([rs_ring_n, ring_n], 2)
        rs_neighbor_corners = torch.cat([rs_neighbor_corners, neighbor_corners], 2)
        # assert rs_ring_n.shape == (num_meshes, num_faces, self.num_samples)
        # assert rs_neighbor_corners.shape == (num_meshes, num_faces, self.num_samples, 3, 3)

        return rs_ring_n, rs_neighbor_corners

class AlphaBetaGamma(nn.Module):
    """
    This class outputs the alpha, beta, and gamma for the three corners in neighbor faces.
    alpha, beta, and gamma are parameters learned by the network independent of corners.
    The sum of alpha, beta, and gamma is constrained to be 1
    """
    def __init__(self, num_faces=500, num_samples=24, device=None):
        """
        Args:
            num_faces: number of faces in each mesh

            num_samples: count of neighbor samples after resampling
        """
        super(AlphaBetaGamma, self).__init__()
        self.num_faces = num_faces
        self.num_samples = num_samples
        self.device = device
        self.alpha = Parameter(torch.rand(self.num_faces, self.num_samples), requires_grad=False)
        self.beta = Parameter(torch.rand(self.num_faces, self.num_samples), requires_grad=False)
        self.gamma = Parameter(torch.rand(self.num_faces, self.num_samples), requires_grad=False)
        self.initialize()

    def initialize(self):
        # The sum of alpha, beta, and gamma is constrained to be 1 with this initialization.
        r1_r2 = torch.rand(2, self.num_faces, self.num_samples)
        r1, r2 = r1_r2
        r1_sqrt = r1.sqrt()
        self.alpha.data = 1.0 - r1_sqrt
        self.beta.data = r1_sqrt * (1.0 - r2)
        self.gamma.data = r1_sqrt * r2

    def forward(self):
        """
        Returns:
            alpha: weights for per-face corner
                   [1, num_faces, num_samples]

            beta: weights for per-face corner
                  [1, num_faces, num_samples]

            gamma: weights for per-face corner
                   [1, num_faces, num_samples]
        """
        device = self.alpha.device
        alpha_beta_gamma = torch.stack([self.alpha, self.beta, self.gamma], 0)
        #
        # #Same barycentric weights per meshes
        alpha_beta_gamma = alpha_beta_gamma.unsqueeze(1)
        # assert alpha_beta_gamma.shape == (3, 1, self.num_faces, self.num_samples)
        #
        # #For points to fall on the surface sum of alpha, beta and gamma must be 1
        # #Directly comparing with ones breaks assertion due to errors in floating points precision
        # in_triangle_constraint = alpha_beta_gamma.sum(dim=0)
        # ones = torch.ones((1, self.num_faces, self.num_samples), device=device)
        # zeros = torch.zeros((1, self.num_faces, self.num_samples), device=device)
        # assert torch.all(torch.eq(abs(ones - in_triangle_constraint).long(), zeros))

        alpha, beta, gamma = alpha_beta_gamma

        return alpha, beta, gamma

class ConvSurface(nn.Module):
    """
    Convolution is performed on the geodesic path between the center of face and
    the points lying on the surface of its neighboring faces.
    """
    def __init__(self, num_faces=500, num_neighbor=3, cfg={}):
        """
        Args:
            num_faces: number of faces in each mesh

            num_neighbor: number of face neighbor in a n-Ring neighborhood

            num_samples_per_neighbor: resampling frequency per neighbor

            rs_mode: mode to resamples faces, edges, and corners in a n-Ring neighborhood.

            num_kernel: dimension of output feature
        """

        super(ConvSurface, self).__init__()
        self.num_neighbor = num_neighbor
        self.num_faces = num_faces
        self.num_samples_per_neighbor = cfg['num_samples_per_neighbor']
        self.rs_mode = cfg['rs_mode']
        self.num_kernel = cfg['num_kernel']
        self.num_samples = self.num_neighbor * self.num_samples_per_neighbor

        if self.rs_mode not in ['Uniform', 'Weighted']:
            raise ValueError('Only Uniform or Weighted resampling of faces are supported!'
                             'Please check configuration file.')

        if self.rs_mode == 'Uniform':
            self.neighbor_resample = NeighborResampleUniform(self.num_samples_per_neighbor)

        if self.rs_mode == 'Weighted':
            self.neighbor_resample = NeighborResampleWeighted(self.num_samples)

        self.alpha_beta_gamma_nonparam = AlphaBetaGamma(self.num_faces, self.num_samples)

        self.directions = nn.Parameter(torch.FloatTensor(3, self.num_kernel))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(self.num_kernel)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.num_samples * self.num_kernel)
        self.directions.data.uniform_(-stdv, stdv)

    def get_neighbor_corners(self, faces, verts, ring_n):
        """
        Helper function to obtain corners in a n-Ring neighborhood
        Args:
            faces: faces in meshes
            [num_meshes, num_faces, 3]

            verts: padded vertices in meshes
            [num_meshes, max(V_n), num_neighbor]

            ring_n: faces in a n-Ring neighborhood of meshes
            [num_meshes, num_faces, num_neighbor]
        Returns:
            neighbor_corners: corners in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor, 3, 3]
        """
        num_meshes = faces.shape[0]
        #For each face, For given neighbor get all corner vertices
        neighbor_faces = faces[torch.arange(num_meshes)[:, None, None], ring_n]
        #For each faces, For given neighbor, For all corner vertex get x,y and z co-ordinates
        neighbor_corners = verts[torch.arange(num_meshes)[:, None, None, None], neighbor_faces]
        return neighbor_corners

    def points_from_surface(self, rs_neighbor_corners, alpha, beta, gamma):
        r"""
        For each face samples points lying on the surface of neighboring faces.
        Points from surface of neighbor faces based on inferred alpha, beta and gamma value of
        face corners. The sum of alpha, beta and gamma is 1 so the points lie in triangular faces.
        Example:
    	              ca1                                 ca1
                     /  \                                / x \
                    /    \                              / x x \
                   /  a   \                            /  a    \
                  /        \                          /   x  x  \
        beta * ca2_ _ _ _ _ca3 * gamma              ca2_x _x x _ca3
                                          ==>
        beta * cb2_ _ _ _ _cb3 * gamma            cb2_ _ _ __ _cb3
                 \          /                        \ x  x x x /
                  \   b    /                          \   b x  /
                   \      /                            \  x x /
                    \    /                              \  x /
                     \  /                                \ x/
                   cb1 * alpha                            cb1

        Args:
            rs_neighbor_corners: resampled n-Ring neighborhood corners
            [num_meshes, num_faces, num_samples, 3, 3]

            alpha: weights for per-face corner
                   [num_meshes, num_faces, num_samples]

            beta: weights for per-face corner
                  [num_meshes, num_faces, num_samples]

            gamma: weights for per-face corner
                   [num_meshes, num_faces, num_samples]
        Returns:
            points_neighbor: sampled point cloud on the surface of neighbor faces
            [num_meshes, num_faces, num_samples, 3]
        """
        num_meshes = rs_neighbor_corners.shape[0]
        num_faces = rs_neighbor_corners.shape[1]
        num_samples = rs_neighbor_corners.shape[2]
        meshes_valid = torch.arange(num_meshes)
        device = rs_neighbor_corners.device

        #Placeholder for sample points
        points_neighbor = torch.zeros((num_meshes, num_faces, num_samples, 3), device=device)

        corner1 = rs_neighbor_corners[:, :, :, 0, :]
        corner2 = rs_neighbor_corners[:, :, :, 1, :]
        corner3 = rs_neighbor_corners[:, :, :, 2, :]

        #Points sampled from neighbor of triangular face such that points lie on the trinagular face
        points_neighbor[meshes_valid] = torch.stack([alpha[:, :, :, None] * corner1,
                                                     beta[:, :, :, None] * corner2,
                                                     gamma[:, :, :, None] * corner3]
                                                    , dim=4).sum(dim=4)

        # assert points_neighbor.shape == (num_meshes, num_faces, num_samples, 3)
        return points_neighbor

    def forward(self, verts, faces, ring_n, centers):
        num_meshes = verts.shape[0]
        num_faces = faces.shape[1]
        num_neighbor = ring_n.shape[2]

        neighbor_corners = self.get_neighbor_corners(faces, verts, ring_n)
        # assert neighbor_corners.shape == (num_meshes, num_faces, num_neighbor, 3, 3)

        rs_neighbor = self.neighbor_resample(ring_n=ring_n,
                                             neighbor_corners=neighbor_corners)

        _, rs_neighbor_corners = rs_neighbor

        alpha, beta, gamma = self.alpha_beta_gamma_nonparam()
        points_neighbor = self.points_from_surface(rs_neighbor_corners, alpha, beta, gamma)
        #Find direction vector from points on neighbor faces to the center of face
        centers = centers.permute(0, 2, 1)

        neighbor_direction = points_neighbor - centers.unsqueeze(2)

        neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
        support_direction_norm = F.normalize(self.directions, dim=0)

        # Sanity Check
        # check_ConvSurface(verts,
        #                   faces,
        #                   centers,
        #                   points_neighbor,
        #                   neighbor_direction,
        #                   neighbor_direction_norm,
        #                   support_direction_norm)

        feature = neighbor_direction_norm @ support_direction_norm
        # assert feature.shape == (num_meshes, num_faces, self.num_samples, self.num_kernel)

        feature = torch.max(feature, dim=2)[0]
        # assert feature.shape == (num_meshes, num_faces, self.num_kernel)

        feature = feature.permute(0, 2, 1)
        feature = self.relu(self.bn(feature))
        return feature
