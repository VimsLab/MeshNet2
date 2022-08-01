import os
import sys
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from config import load_config
from data import rpartition_dataset, load_partition
from models import MeshNet2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test(model, device, test_loader, criterion):
    print('Evaluating on ' + str(len(test_loader.dataset)) + ' meshes...')
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    for collated_dict in test_loader:
        ring_1 = torch.stack(collated_dict['ring_1']).to(device)
        ring_2 = torch.stack(collated_dict['ring_2']).to(device)
        ring_3 = torch.stack(collated_dict['ring_3']).to(device)

        targets = torch.stack(collated_dict['target']).to(device)
        meshes = collated_dict['meshes'].to(device)

        #Check for empty meshes
        if meshes.isempty():
            raise ValueError("Meshes are empty.")
        #Check valid meshes equal batch size
        num_meshes = len(meshes.valid)
        #Check number of faces equal num_faces
        num_faces = meshes.num_faces_per_mesh().max().item()

        # Each vertex is a point with x,y and z co-ordinates
        verts = meshes.verts_padded()
        # Normals for scaled vertices
        normals = meshes.faces_normals_padded()
        # Each face contains index of its corner vertex
        faces = meshes.faces_padded()

        if not torch.isfinite(verts).all():
            raise ValueError("Mesh vertices contain nan or inf.")
        if not torch.isfinite(normals).all():
            raise ValueError("Mesh normals contain nan or inf.")

        corners = verts[torch.arange(num_meshes)[:, None, None], faces.long()]
        centers = torch.sum(corners, axis=2)/3
        # Each mesh face has one center
        assert centers.shape == (num_meshes, num_faces, 3)
        # Each face only has 3 corners
        assert corners.shape == (num_meshes, num_faces, 3, 3)

        assert ring_1.shape == (num_meshes, num_faces, 3)

        assert ring_2.shape == (num_meshes, num_faces, 6)

        assert ring_3.shape == (num_meshes, num_faces, 12)

        centers = centers.permute(0, 2, 1)
        normals = normals.permute(0, 2, 1)

        verts = Variable(verts)
        faces = Variable(faces)
        centers = Variable(centers)
        normals = Variable(normals)

        ring_1 = Variable(ring_1)
        ring_2 = Variable(ring_2)
        ring_3 = Variable(ring_3)

        targets = Variable(targets.cuda())

        with torch.no_grad():
            outputs = model(verts=verts,
                            faces=faces,
                            centers=centers,
                            normals=normals,
                            ring_1=ring_1,
                            ring_2=ring_2,
                            ring_3=ring_3)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * centers.size(0)
            # if preds[0] == targets[0]:
            #     running_corrects += 1
            running_corrects += torch.sum(preds == targets.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects / len(test_loader.dataset)
    print('Test loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc))
    return test_acc

if __name__ == '__main__':
    # os.environ['PYTHONHASHSEED'] = str(0)
    # np.random.seed(0)
    # random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(False)

    if len(sys.argv) <= 1 or len(sys.argv) > 3:
        print('Use: python robustness.py <arch> <dataset> <section>')
        print('<dataset> can be one of the following: ')
        print('rSHREC11, rMSB')
        print('<section> should be used in case of SHREC11 or MSB datasets.')
        print('For example:')
        print('python test.py SHREC11 16-04_A')
        exit(0)
    elif len(sys.argv) == 3:
        dataset = sys.argv[1]
        section = sys.argv[2]
        print('Dataset: ' + dataset + ', Section: ' + section)
        cfg = load_config('config/{0}.yaml'.format(dataset))
        cfg_dataset = cfg[section]

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
    device = torch.device("cuda:0")

    robust_acc = []
    for max_rot_ang_deg in range(10, 370, 10):
        # Load data
        test_data = rpartition_dataset(dataset=dataset,
                                       section=section,
                                       partition='test',
                                       augment=cfg['augment'],
                                       max_rot_ang_deg=max_rot_ang_deg)
        if len(test_data) != cfg_dataset['num_test']:
            raise ValueError('Train/Test split incorrect. Check again!')

        num_test = cfg_dataset['num_test']

        print('#' * 60)
        print('Number of meshes in test set: ' + str(num_test))

        num_cls = cfg['num_cls']
        print('Number of classes: ' + str(num_cls))

        num_faces = cfg['num_faces']
        print('Number of faces: ' + str(num_faces))

        batch_size = cfg['batch_size']

        print('Data loader batch size: ' + str(batch_size))
        print('#' * 60)

        test_loader = load_partition(partition_data=test_data, batch_size=batch_size)
        #Load model, setup loss function, load pre-trained weights
        model = MeshNet2(cfg=cfg, num_faces=num_faces, num_cls=num_cls, pool_rate=cfg['pool_rate'])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        model.load_state_dict(torch.load(cfg_dataset['ckpt_root'] + '/MeshNet2_best.pkl'))
        model.eval()
        criterion = nn.CrossEntropyLoss()
        print('#' * 60)

        test_accuracy = test(model=model, device=device, test_loader=test_loader, criterion=criterion)
        robust_acc.append(test_accuracy)

    print('Robust Accuracy : ' + str(sum(robust_acc)/len(robust_acc)))
    print(len(robust_acc))
