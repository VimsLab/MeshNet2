""" MeshNet2 Layers """
import torch
import torch.nn as nn

class MaxPoolFaceFeature(nn.Module):
    r"""
    Retrives maximum channel value from amonng the faces and its n-Ring neighborhood.
    E.g: Let face "f" and its 1-ring neighbors "n1", "n2", and, "n3" have channels
    "cf", "cn1", "cn2", "cn3" as shown below.
             _ _          _ _          _ _
            |   |        |   |        |   |
          _ |cn1|_       |cf |      _ |cn2| _
          \ |_ _| /      |_ _|      \ |_ _| /
           \ n1  /      /  f  \      \ n2  /
            \   /      /_ _ _ _\      \   /
             \ /          _ _          \ /
                         |   |
                       _ |cn3| _
                       \ |_ _| /
                        \ n3  /
                         \   /
                          \ /

    Then, MaxPoolFaceFeature retrives max(cf, cn1, cn2, cn3) for f and re-assigns it to f.
    """
    def __init__(self, in_channel, num_neighbor=3):
        """
        Args:
            in_channel: number of channels in feature

            num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(MaxPoolFaceFeature, self).__init__()
        self.in_channel = in_channel
        self.num_neighbor = num_neighbor

    def forward(self, fea, ring_n):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood.
            [num_meshes, num_faces, num_neighbor]

        Returns:
            max_fea: maximum channel value from amonng the faces and its n-Ring neighborhood.
            [num_meshes, in_channel, num_faces]
        """
        num_meshes, num_channels, num_faces = fea.size()
        # assert ring_n.shape == (num_meshes, num_faces, self.num_neighbor)
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)

        # Gather features at face neighbors
        fea = fea.unsqueeze(3)
        ring_n = ring_n.unsqueeze(1)
        ring_n = ring_n.expand(num_meshes, num_channels, num_faces, -1)

        neighbor_fea = fea[
            torch.arange(num_meshes)[:, None, None, None],
            torch.arange(num_channels)[None, :, None, None],
            ring_n
        ]

        neighbor_fea = neighbor_fea.squeeze(4)
        # Concatenate gathered neighbor features to face_feature, and then find the max
        fea = torch.cat([fea, neighbor_fea], 3)
        # assert fea.shape == (num_meshes, num_channels, num_faces, self.num_neighbor + 1)

        max_fea = torch.max(fea, dim=3).values
        # assert max_fea.shape == (num_meshes, self.in_channel, num_faces)

        return max_fea

class ConvFace(nn.Module):
    r"""
    Convolves the channel values of the faces with its n-Ring neighborhood.
    E.g: Let face "f" and its 1-ring neighbors "n1", "n2", "n3" have channels "cf",
    "cn1", "cn2", "cn3" as shown below.
             _ _          _ _          _ _
            |   |        |   |        |   |
          _ |cn1|_       |cf |      _ |cn2| _
          \ |_ _| /      |_ _|      \ |_ _| /
           \ n1  /      /  f  \      \ n2  /
            \   /      /_ _ _ _\      \   /
             \ /          _ _          \ /
                         |   |
                       _ |cn3| _
                       \ |_ _| /
                        \ n3  /
                         \   /
                          \ /

    Then, for f, ConvFace computes sum(cf, cn1, cn2, cn3) and passes it along to
    Conv1D to perform convolution.
    """
    def __init__(self, in_channel, out_channel, num_neighbor):
        """
        Args:
            in_channel: number of channels in feature

            out_channel: number of channels produced by convolution

            num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(ConvFace, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_neighbor = num_neighbor
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(self.in_channel, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(),
        )

    def forward(self, fea, ring_n):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

        Returns:
            conv_fea: features produced by convolution of faces with its
            n-Ring neighborhood features
            [num_meshes, out_channel, num_faces]
        """
        num_meshes, num_channels, num_faces = fea.size()
        # assert ring_n.shape == (num_meshes, num_faces, self.num_neighbor)
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)

        # Gather features at face neighbors
        fea = fea.unsqueeze(3)
        ring_n = ring_n.unsqueeze(1)
        ring_n = ring_n.expand(num_meshes, num_channels, num_faces, -1)

        neighbor_fea = fea[
            torch.arange(num_meshes)[:, None, None, None],
            torch.arange(num_channels)[None, :, None, None],
            ring_n
        ]

        neighbor_fea = neighbor_fea.squeeze(4)
        # Concatenate gathered neighbor features to face_feature, and then find the sum
        fea = torch.cat([fea, neighbor_fea], 3)
        # assert fea.shape == (num_meshes, num_channels, num_faces, self.num_neighbor + 1)

        fea = torch.sum(neighbor_fea, 3)
        # assert fea.shape == (num_meshes, num_channels, num_faces)

        # Convolve
        conv_fea = self.concat_mlp(fea)
        # assert conv_fea.shape == (num_meshes, self.out_channel, num_faces)

        return conv_fea

class ConvFaceBlock(nn.Module):
    """
    Multiple ConvFaceBlock layers create a MeshBlock.
    ConvFaceBlock is comprised of ConvFace layers.
    First ConvFace layer convolves on in_channel to produce "128" channels.
    Second ConvFace convolves these "128" channels to produce "growth factor" channels.
    These features get concatenated to the original input feature to produce
    "in_channel + growth_factor" channels.
    """
    def __init__(self, in_channel, growth_factor, num_neighbor):
        """
        Args:
        in_channel: number of channels in feature

        growth_factor: number of channels to increase in_channel by

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(ConvFaceBlock, self).__init__()
        self.in_channel = in_channel
        self.growth_factor = growth_factor
        self.num_neighbor = num_neighbor
        self.conv_face_1 = ConvFace(self.in_channel, 128, self.num_neighbor)
        self.conv_face_2 = ConvFace(128, self.growth_factor, self.num_neighbor)

    def forward(self, fea, ring_n):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

        Returns:
            conv_block_fea: features produced by ConvFaceBlock layer
            [num_meshes, in_channel + growth_factor, num_faces]
        """
        num_meshes, num_channels, num_faces = fea.size()
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)
        fea_copy = fea
        fea = self.conv_face_1(fea, ring_n)
        fea = self.conv_face_2(fea, ring_n)
        conv_block_fea = torch.cat([fea_copy, fea], 1)
        # assert conv_block_fea.shape == (num_meshes, self.in_channel + self.growth_factor, num_faces)

        return conv_block_fea

class MeshBlock(nn.ModuleDict):
    """
    Multiple MeshBlock layers create MeshNet2.
    MeshBlock is comprised of several ConvFaceBlock layers.
    """
    def __init__(self, in_channel, num_neighbor, num_block, growth_factor):
        """
        in_channel: number of channels in feature

        growth_factor: number of channels a single ConvFaceBlock increase in_channel by

        num_block: number of ConvFaceBlock layers in a single MeshBlock

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(MeshBlock, self).__init__()
        self.in_channel = in_channel
        self.growth_factor = growth_factor
        self.num_block = num_block
        self.num_neighbor = num_neighbor

        for i in range(0, num_block):
            layer = ConvFaceBlock(in_channel=in_channel,
                                  growth_factor=growth_factor,
                                  num_neighbor=num_neighbor)
            in_channel += growth_factor
            self.add_module('meshblock%d' % (i + 1), layer)

    def forward(self, fea, ring_n):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

        Returns:
            fea: features produced by MeshBlock layer
            [num_meshes, in_channel + growth_factor * num_block, num_faces]
        """
        num_meshes, num_channels, num_faces = fea.size()
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)
        for _, layer in self.items():
            fea = layer(fea, ring_n)
        # out_channel = self.in_channel + self.growth_factor * self.num_block
        # assert fea.shape == (num_meshes, out_channel, num_faces)
        return fea
