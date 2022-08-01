"""
Data loader for 3D-FUTURE data set
"""
import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data

class FUTURE3D(data.Dataset):
    """ 3D-FUTURE dataset """
    def __init__(self,
                 data_root='',
                 partition='',
                 category_to_idx_map={},
                 augment=''):
        """
        Args:
            data_root: root directory where the 3D-FUTURE dataset is stored
            partition: train or test partition of data
            augment: type of augmentation, e.g: scale_verts
        """
        self.data_root = data_root
        self.partition = partition
        self.category_to_idx_map = category_to_idx_map
        self.augment = augment
        self.data = []

        for category in os.listdir(self.data_root):
            category_index = self.category_to_idx_map[category]
            category_root = osp.join(osp.join(self.data_root, category), partition)
            for filename in os.listdir(category_root):
                if filename.endswith('.npz'):
                    self.data.append((osp.join(category_root, filename), category_index))

    def __getitem__(self, i):
        # Read mesh properties for .npz files
        path, target = self.data[i]
        mesh = np.load(path)
        faces = mesh['faces']
        verts = mesh['verts']
        ring_1 = mesh['ring_1']
        ring_2 = mesh['ring_2']
        ring_3 = mesh['ring_3']

        # Convert to tensor
        faces = torch.from_numpy(faces).long()
        verts = torch.from_numpy(verts).float()
        ring_1 = torch.from_numpy(ring_1).long()
        ring_2 = torch.from_numpy(ring_2).long()
        ring_3 = torch.from_numpy(ring_3).long()
        target = torch.tensor(target, dtype=torch.long)

        # Perform augmentation during training
        if self.partition == 'train' and self.augment == 'scale_verts':
            verts = verts.numpy()
            # Scale verticies during training
            for v in range(verts.shape[1]):
                verts[:, v] = verts[:, v] * np.random.normal(1, 0.1)

            verts = torch.from_numpy(verts).float()

        # Dictionary for collate_batched_meshes
        collated_dict = {
            'faces': faces,
            'verts': verts,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3,
            'target': target
        }

        return collated_dict

    def __len__(self):
        return len(self.data)
