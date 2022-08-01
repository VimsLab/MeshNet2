import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from config import load_config
from data import partition_dataset, load_partition
from models import MeshNet2
from models import check_Input

def train(model, device, train_loader, criterion, optimizer):
    """
    Args:
    model: model to train
    device: torch device
    train_loader: data loader for training data
    criterion: categorical cross-entropy loss function
    optimizer: optimizer
    """
    print('Training on ' + str(len(train_loader.dataset)) + ' meshes...')
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for collated_dict in train_loader:
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

        # Sanity Check
        # check_Input(verts, faces, centers, corners, targets)

        optimizer.zero_grad()
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

        with torch.set_grad_enabled(True):
            outputs = model(verts=verts,
                            faces=faces,
                            centers=centers,
                            normals=normals,
                            ring_1=ring_1,
                            ring_2=ring_2,
                            ring_3=ring_3)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * centers.size(0)
            running_corrects += torch.sum(preds == targets.data)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects / len(train_loader.dataset)
    print('Train loss: {:.4f}, Train Accuracy: {:.4f}'.format(train_loss, train_acc))


def validate(model, device, val_loader, criterion):
    """
    Args:
    model: model to train
    device: torch device
    val_loader: data loader for validation data
    criterion: categorical cross-entropy loss function
    """
    print('Evaluating on ' + str(len(val_loader.dataset)) + ' meshes...')
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    for collated_dict in val_loader:
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
            running_corrects += torch.sum(preds == targets.data)

    val_loss = running_loss / len(val_loader.dataset)
    _val_acc = running_corrects / len(val_loader.dataset)
    print('Validation loss: {:.4f}, Validation Accuracy: {:.4f}'.format(val_loss, _val_acc))
    return _val_acc

if __name__ == '__main__':
    if len(sys.argv) <= 1 or len(sys.argv) > 3:
        print('Use: python train_val.py <arch> <dataset> <section>')
        print('<dataset> can be one of the following: ')
        print('CUBES, SHREC11, FUTURE3D, ModelNet10, ModelNet40, MSB')
        print('<section> should be used in case of SHREC11 or MSB datasets.')
        print('For example:')
        print('python train_val.py SHREC11 16-04_A')
        print('python train_val.py ModelNet40')
        exit(0)
    elif len(sys.argv) == 2:
        dataset = sys.argv[1]
        section = ''
        print('Dataset: ' + dataset)
        cfg = load_config('config/{0}.yaml'.format(dataset))
        cfg_dataset = cfg
    elif len(sys.argv) == 3:
        dataset = sys.argv[1]
        section = sys.argv[2]
        print('Dataset: ' + dataset + ', Section: ' + section)
        cfg = load_config('config/{0}.yaml'.format(dataset))
        cfg_dataset = cfg[section]

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
    device = torch.device("cuda:0")

    # Load data
    train_data = partition_dataset(dataset=dataset, section=section, partition='train', augment=cfg['augment'])
    val_data = partition_dataset(dataset=dataset, section=section, partition='test', augment=False)
    if len(train_data) != cfg_dataset['num_train'] or  len(val_data) != cfg_dataset['num_val']:
        raise ValueError('Train/Test split incorrect. Check again!')

    num_train = cfg_dataset['num_train']
    num_val = cfg_dataset['num_val']

    print('#' * 60)
    print('Number of meshes in train set: ' + str(num_train))
    print('Number of meshes in validation set: ' + str(num_val))

    num_cls = cfg['num_cls']
    print('Number of classes: ' + str(num_cls))

    num_faces = cfg['num_faces']
    print('Number of faces: ' + str(num_faces))

    batch_size = cfg['batch_size']

    print('Data loader batch size: ' + str(batch_size))
    train_loader = load_partition(partition_data=train_data, batch_size=batch_size)
    val_loader = load_partition(partition_data=val_data, batch_size=batch_size)
    print('#' * 60)

    #Load model, setup loss function, optimizer, and scheduler
    model = MeshNet2(cfg=cfg, num_faces=num_faces, num_cls=num_cls, pool_rate=cfg['pool_rate'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))

    print('#' * 60)
    drop_lr = False
    max_epoch = cfg['max_epoch']
    if 'milestones' in cfg:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
        drop_lr = True
    print('Training for {0} epochs... '.format(str(max_epoch)))

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']+1):
        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        train(model=model,
              device=device,
              train_loader=train_loader,
              criterion=criterion,
              optimizer=optimizer)

        if drop_lr:
            scheduler.step()
        print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))

        val_acc = validate(model=model,
                           device=device,
                           val_loader=val_loader,
                           criterion=criterion)

        if val_acc >= best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = val_acc

        if not os.path.exists(cfg_dataset['ckpt_root']):
            os.makedirs(cfg_dataset['ckpt_root'])

        model_wts = copy.deepcopy(model.state_dict())
        torch.save(model_wts, cfg_dataset['ckpt_root']  + '/{0}.pkl'.format(epoch))

    torch.save(best_model_wts, cfg_dataset['ckpt_root'] + '/MeshNet2_best.pkl')
