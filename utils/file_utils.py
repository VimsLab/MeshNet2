"""
Collection of utility function for reading file in batches
"""
import os
import numpy as np
import subprocess

def fpath(dir_name):
    """
    Return all obj file in a directory

    Args:
        dir_name: root path to obj files

    Returns:
        f_path: list of obj files paths
    """
    f_path = []
    for root, dirs, files in os.walk(dir_name, topdown=False):
        for f in files:
            if f.endswith('.obj'):
                if os.path.exists(os.path.join(root, f)):
                    f_path.append(os.path.join(root, f))
    return f_path
