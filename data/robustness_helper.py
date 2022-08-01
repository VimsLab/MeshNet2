import os
import os.path as osp
import json
from typing import List, Dict
import torch
from pytorch3d.structures import Meshes
from data import rSHREC11, rMSB

def rpartition_dataset(dataset='', section='', partition='', augment=False, max_rot_ang_deg=360):
    r"""
    Helper function to get train or test partition from dataset.

    Args:
        dataset: dataset can be either 'SHREC11' or, 'MSB'
        section: should be used in case of SHREC11 or MSB datasets
        partition: test partition of data
        augment: boolean value indicating whether to perform augmentation or not

    Returns:
        partition_data: data from train or test partition of dataset
    """

    dataset_options = ['SHREC11', 'MSB']

    if partition not in ['test']:
        raise ValueError('Partition can only be test!')

    if dataset not in dataset_options:
        raise ValueError('Only dataset options are : SHREC11 and MSB')

    if max_rot_ang_deg < 0 or max_rot_ang_deg > 360:
        raise ValueError('Rotation angle should be between 0 to 360.')

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

        partition_data = rSHREC11(data_root=data_root,
                                  partition=partition,
                                  category_to_idx_map=category_to_idx_map,
                                  augment=augment,
                                  max_rot_ang_deg=max_rot_ang_deg)

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

        partition_data = rMSB(data_root=data_root,
                              partition=partition,
                              category_to_idx_map=category_to_idx_map,
                              augment=augment,
                              max_rot_ang_deg=max_rot_ang_deg)
    return partition_data
