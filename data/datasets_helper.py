import os
import os.path as osp
import json
from typing import List, Dict
import torch
from pytorch3d.structures import Meshes
from data import SHREC11, CUBES, ModelNet10, FUTURE3D, ModelNet40, MSB

def partition_dataset(dataset='', section='', partition='', augment=False):
    r"""
    Helper function to get train or test partition from dataset.

    Args:
        dataset: dataset can be either 'CUBES', 'SHREC11',
        'FUTURE3D', 'ModelNet10', 'ModelNet40', or, 'MSB'
        section: should be used in case of SHREC11 or MSB datasets
        partition: train or test partition of data
        augment: boolean value indicating whether to perform augmentation or not

    Returns:
        partition_data: data from train or test partition of dataset
    """

    dataset_options = [
        'CUBES', 'SHREC11', 'FUTURE3D',
        'ModelNet10', 'ModelNet40', 'MSB']

    if partition not in ['train', 'test']:
        raise ValueError('Partition can only either be train or test!')

    if dataset not in dataset_options:
        raise ValueError('Only dataset options are : CUBES, SHREC11, FUTURE3D',
                         'ModelNet10, ModelNet40, and, MSB.')

    elif dataset == 'ModelNet40':
        cwd = os.getcwd()
        data_root = cwd + '/datasets/ModelNet40/' + section
        if not osp.exists(data_root):
            raise ValueError('Root directory {0} does not exists!'.format(data_root))

        label_root = cwd + '/data/labels/ModelNet40.json'
        with open(label_root) as json_file:
            category_to_idx_map = json.loads(json_file.read())
        if len(category_to_idx_map) != 40:
            raise ValueError('ModelNet40 has 40 classes!')

        if augment:
            augment = 'scale_verts'

        partition_data = ModelNet40(data_root=data_root,
                                    partition=partition,
                                    category_to_idx_map=category_to_idx_map,
                                    augment=augment)

    elif dataset == 'FUTURE3D':
        cwd = os.getcwd()
        data_root = cwd + '/datasets/FUTURE3D/' + section
        if not osp.exists(data_root):
            raise ValueError('Root directory {0} does not exists!'.format(data_root))

        label_root = cwd + '/data/labels/FUTURE3D.json'
        with open(label_root) as json_file:
            category_to_idx_map = json.loads(json_file.read())
        if len(category_to_idx_map) != 34:
            raise ValueError('3D-FUTURE has 34 classes!')

        if augment:
            augment = 'scale_verts'

        partition_data = FUTURE3D(data_root=data_root,
                                  partition=partition,
                                  category_to_idx_map=category_to_idx_map,
                                  augment=augment)

    elif dataset == 'ModelNet10':
        cwd = os.getcwd()
        data_root = cwd + '/datasets/ModelNet10/' + section
        if not osp.exists(data_root):
            raise ValueError('Root directory {0} does not exists!'.format(data_root))

        label_root = cwd + '/data/labels/ModelNet10.json'
        with open(label_root) as json_file:
            category_to_idx_map = json.loads(json_file.read())
        if len(category_to_idx_map) != 10:
            raise ValueError('ModelNet10 has 10 classes!')

        if augment:
            augment = 'scale_verts'

        partition_data = ModelNet10(data_root=data_root,
                                    partition=partition,
                                    category_to_idx_map=category_to_idx_map,
                                    augment=augment)

    elif dataset == 'CUBES':
        cwd = os.getcwd()
        data_root = cwd + '/datasets/CUBES/' + section
        if not osp.exists(data_root):
            raise ValueError('Root directory {0} does not exists!'.format(data_root))

        label_root = cwd + '/data/labels/CUBES.json'
        with open(label_root) as json_file:
            category_to_idx_map = json.loads(json_file.read())
        if len(category_to_idx_map) != 22:
            raise ValueError('CUBES has 22 classes!')

        if augment:
            augment = 'scale_verts'

        partition_data = CUBES(data_root=data_root,
                               partition=partition,
                               category_to_idx_map=category_to_idx_map,
                               augment=augment)

    elif dataset == 'SHREC11':
        if section not in ['16-04_A', '16-04_B', '16-04_C', '10-10_A', '10-10_B', '10-10_C']:
            raise ValueError('Invalid section of SHREC11 data! Valid choices are: ',
                             '16-04_A, 16-04_B, 16-04_C, 10-10_A, 10-10_B, 10-10_C')

        cwd = os.getcwd()
        data_root = cwd + '/datasets/SHREC11/' + section
        if not osp.exists(data_root):
            raise ValueError('Root directory {0} does not exists!'.format(data_root))

        label_root = cwd + '/data/labels/SHREC11.json'
        with open(label_root) as json_file:
            category_to_idx_map = json.loads(json_file.read())
        if len(category_to_idx_map) != 30:
            raise ValueError('SHREC11 has 30 classes!')

        if augment:
            augment = 'rotate'

        partition_data = SHREC11(data_root=data_root,
                                 partition=partition,
                                 category_to_idx_map=category_to_idx_map,
                                 augment=augment)

    elif dataset == 'MSB':
        if section not in ['MSB_1', 'MSB_2', 'MSB_3', 'MSB_4', 'MSB_5',
                           'MSB_6', 'MSB_7', 'MSB_8', 'MSB_9', 'MSB_10']:
            raise ValueError('Invalid section of SHREC11 data! Valid choices are: ',
                             'MSB_1, MSB_2, MSB_3, MSB_4, MSB_5',
                             'MSB_6, MSB_7, MSB_8, MSB_9, MSB_10')

        cwd = os.getcwd()
        data_root = cwd + '/datasets/MSB/' + section
        if not osp.exists(data_root):
            raise ValueError('Root directory {0} does not exists!'.format(data_root))

        label_root = cwd + '/data/labels/MSB.json'
        with open(label_root) as json_file:
            category_to_idx_map = json.loads(json_file.read())
        if len(category_to_idx_map) != 19:
            raise ValueError('MSB has 19 classes!')

        if augment:
            augment = 'rotate'

        partition_data = MSB(data_root=data_root,
                             partition=partition,
                             category_to_idx_map=category_to_idx_map,
                             augment=augment)

    return partition_data


def collate_batched_meshes(batch: List[Dict]):
    """
    Take a list of objects in the form of dictionaries and merge them
    into a single dictionary. This function is used with a Dataset
    object to create a torch.utils.data.Dataloader which directly
    returns Meshes objects.

    Args:
        batch: List of dictionaries containing information about objects
        in the dataset.

    Returns:
        collated_dict: Dictionary of collated lists. If batch contains both
        verts and faces, a collated mesh batch is also returned.
    """
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]
    collated_dict['meshes'] = None

    if {'verts', 'faces'}.issubset(collated_dict.keys()):
        collated_dict['meshes'] = Meshes(
            verts=collated_dict['verts'],
            faces=collated_dict['faces'],
        )

    collated_dict.pop('verts')
    collated_dict.pop('faces')

    return collated_dict

def load_partition(partition_data, batch_size):

    partition_loader = torch.utils.data.DataLoader(
        partition_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        collate_fn=collate_batched_meshes,
        drop_last=False
    )

    return partition_loader
