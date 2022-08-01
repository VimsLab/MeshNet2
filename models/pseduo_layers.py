""" MeshNet2 Psuedo Layers """
import torch
import torch.nn as nn

class PsuedoConvFace(nn.Module):
    def __init__(self, in_channel, out_channel, num_neighbor):
        """
        Args:
            in_channel: number of channels in feature

            out_channel: number of channels produced by convolution

            num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(PsuedoConvFace, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_neighbor = num_neighbor

        self.concat_mlp = nn.Sequential(
            nn.Conv1d(self.in_channel, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(),
        )

    def forward(self, fea, ring_n, pool_idx):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

            pool_idx: indices of faces to be considered for spatial pooling
            [num_faces]//2 OR [num_faces]//4

        Returns:
            conv_fea: features produced by convolution of faces with its
            n-Ring neighborhood features
            [num_meshes, out_channel, num_faces]
        """
        num_meshes, num_channels, _ = fea.size()
        _, num_faces, _ = ring_n.size()
        # assert ring_n.shape == (num_meshes, num_faces, self.num_neighbor)

        # Gather features at face neighbors only at pool_idx
        fea = fea.unsqueeze(3)
        ring_n = ring_n.unsqueeze(1)
        ring_n = ring_n.expand(num_meshes, num_channels, num_faces, -1)

        neighbor_fea = fea[
            torch.arange(num_meshes)[:, None, None, None],
            torch.arange(num_channels)[None, :, None, None],
            ring_n
        ]
        neighbor_fea = neighbor_fea.squeeze(4)

        # Pool input feature only at pool_idx
        # Pooling here occurs at the spatial dimension
        fea = fea[:, :, pool_idx, :]

        # Concatenate gathered neighbor features to face_feature, and then find the sum
        fea = torch.cat([fea, neighbor_fea], 3)
        # assert fea.shape == (num_meshes, num_channels, num_faces, self.num_neighbor + 1)
        fea = torch.sum(fea, 3)

        conv_fea = self.concat_mlp(fea)
        # assert conv_fea.shape == (num_meshes, self.out_channel, num_faces)

        return conv_fea

class PsuedoConvFaceBlock(nn.Module):
    """
    Multiple PsuedoConvFaceBlock layers create a PsuedoMeshBlock.
    PsuedoConvFaceBlock is comprised of PsuedoConvFace layers.
    First PsuedoConvFace layer convolves on in_channel to produce "128" channels.
    Second PsuedoConvFace convolves these "128" channels to produce "growth factor" channels.
    These features get concatenated to the original input feature to produce
    "in_channel + growth_factor" channels.
    Note: The original mesh dimensions are maintained for gathering the neighbor features but
    the operations get perfomed only on the pooling indices.
    """
    def __init__(self, in_channel, growth_factor, num_neighbor):
        """
        Args:
        in_channel: number of channels in feature

        growth_factor: number of channels to increase in_channel by

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(PsuedoConvFaceBlock, self).__init__()
        self.in_channel = in_channel
        self.growth_factor = growth_factor
        self.num_neighbor = num_neighbor
        self.pconv_face_1 = PsuedoConvFace(in_channel, 128, num_neighbor)
        self.pconv_face_2 = PsuedoConvFace(128, growth_factor, num_neighbor)

    def forward(self, fea, ring_n, pool_idx):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

            pool_idx: indices of faces to be considered for spatial pooling
            [num_faces]//2 OR [num_faces]//4

        Returns:
            conv_block_fea: features produced by ConvFaceBlock layer
            [num_meshes, in_channel + growth_factor, num_faces]
        """
        fea_copy = fea
        device = fea.device
        num_meshes, num_channels, num_faces = fea.size()
        # assert ring_n.shape == (num_meshes, pool_idx.shape[0], self.num_neighbor)
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)

        n = torch.arange(num_meshes)[:, None, None]
        p = pool_idx[None, None, :]

        # Convolve
        fea = self.pconv_face_1(fea, ring_n, pool_idx)
        # assert fea.shape == (num_meshes, fea.shape[1], pool_idx.shape[0])
        # Create placeholder for tensor re-assignment
        fea_placeholder = torch.zeros((num_meshes, fea.shape[1], num_faces), device=device)
        c = torch.arange(fea.shape[1])[None, :, None]
        # Assign values from fea to fea_placeholder at pooling indicies
        # Values at non pooling indices will be zero
        fea_placeholder[n, c, p] = fea
        # assert fea_placeholder.shape == (num_meshes, fea.shape[1], num_faces)

        # Convolve
        fea = self.pconv_face_2(fea_placeholder, ring_n, pool_idx)
        # Create placeholder for tensor re-assignment
        fea_placeholder = torch.zeros((num_meshes, fea.shape[1], num_faces), device=device)
        c = torch.arange(fea.shape[1])[None, :, None]
        # Assign values from fea to fea_placeholder at pooling indicies
        # Values at non pooling indices will be zero
        fea_placeholder[n, c, p] = fea
        # assert fea_placeholder.shape == (num_meshes, fea.shape[1], num_faces)

        conv_block_fea = torch.cat([fea_copy, fea_placeholder], 1)
        # assert conv_block_fea.shape == (num_meshes, self.in_channel + self.growth_factor, num_faces)

        return conv_block_fea

class PsuedoMeshBlock(nn.ModuleDict):
    """
    Multiple PsuedoMeshBlock layers create MeshNet2.
    PsuedoMeshBlock is comprised of several PsuedoConvFaceBlock layers.
    """
    def __init__(self, in_channel, num_block, growth_factor, num_neighbor):
        """
        in_channel: number of channels in feature

        growth_factor: number of channels a single ConvFaceBlock increase in_channel by

        num_block: number of ConvFaceBlock layers in a single MeshBlock

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(PsuedoMeshBlock, self).__init__()
        for i in range(0, num_block):
            layer = PsuedoConvFaceBlock(in_channel, growth_factor, num_neighbor)
            in_channel += growth_factor
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, fea, ring_n, pool_idx):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

            pool_idx: indices of faces to be considered for spatial pooling
            [num_faces]//2 OR [num_faces]//4

        Returns:
            fea: features produced by MeshBlock layer
            [num_meshes, in_channel + growth_factor * num_block, num_faces]
        """
        ring_n = ring_n[:, pool_idx, :]
        for _, layer in self.items():
            fea = layer(fea, ring_n, pool_idx)
        return fea
