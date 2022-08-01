import os
import os.path as osp
import yaml

def load_config(path_config = ''):
    '''
        Loads global configurations required to run the project

        Args:
            path_config: path to the yaml configuration file

        Returns:
            cfg: key value pairs of configurations required to run the project
    '''
    if not osp.exists(path_config):
        raise Exception('Configuration file does not exist!')

    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg
