import os
import logging
import yaml
import time
from pathlib import Path
from typing import Dict, List, Union
# Better CPU Utilization
os.environ['OMP_NUM_THREADS'] = str(int(os.cpu_count()))
os.chdir('/home/shubham/Academics/Thesis/openfl/openfl-tutorials/interactive_api/PyTorch_MedMNIST_2D')
os.getcwd()
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
from pprint import pprint

import torch
import medmnist

# Train/test options
NUM_EPOCHS = 3
BATCH_SIZE = 64
DEVICE = 'cuda'

# Dataset
DATASET_NAME = 'bloodmnist'
DATASET_PATH = './data'
ds_info = medmnist.INFO[DATASET_NAME]
pprint(ds_info)

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

from envoy.medmnist_shard_descriptor import MedMNISTShardDescriptor

# Download raw numpy dataset
sd = MedMNISTShardDescriptor(datapath=DATASET_PATH, dataname=DATASET_NAME)
(x_train, y_train), (x_test, y_test) = sd.load_data()

# Visualize few samples
n_img = 5
train_samples = x_train[:n_img*n_img]

label2str = list(ds_info['label'].values())

fig, ax = plt.subplots(n_img, n_img, figsize=(8, 8))

for k in range(len(train_samples)):
    i = k // n_img
    j = k % n_img
    img = train_samples[k]
    label = np.squeeze(y_train[k])
    ax[i, j].imshow(Image.fromarray(img))
    ax[i, j].title.set_text(label2str[label][:9])
    ax[i, j].axis('off')
plt.suptitle(DATASET_NAME)
fig.subplots_adjust(wspace=0.03, hspace=0.3)


class MedMNISTDataset(Dataset):
    """MedMNIST dataset class"""

    def __init__(self, x, y, data_type: str = 'train') -> None:
        """Initialize MedMNISTDataset."""
        self.x, self.y = x, y
        self.data_type = data_type

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        """Return the len of the dataset."""
        return len(self.x)


class TransformDataset(Dataset):
    """Apply transforms to each element of dataset"""

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]

        if self.target_transform:
            label = self.target_transform(label)

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        
        return img, label


transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.], std=[1.0])])

train_ds = TransformDataset(MedMNISTDataset(x=x_train, y=y_train),
                              transform=transform)
train_dl = DataLoader(train_ds,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=8)

test_ds = TransformDataset(MedMNISTDataset(x=x_test, y=y_test),
                             transform=transform)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=8)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=3),
                                    nn.BatchNorm2d(16), nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3),
                                    nn.BatchNorm2d(16), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=3),
                                    nn.BatchNorm2d(64), nn.ReLU())

        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),
                                    nn.BatchNorm2d(64), nn.ReLU())

        self.layer5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 128), 
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


test_model = Net(in_channels=ds_info['n_channels'],
                 num_classes=len(ds_info['label']))
print(test_model)
print('Total Parameters:',
      sum([torch.numel(p) for p in test_model.parameters()]))
print('Trainable Parameters:',
      sum([torch.numel(p) for p in test_model.parameters() if p.requires_grad]))

def train(model, train_loader, optimizer, device, criterion, task):
    model.train()
    model = model.to(device)

    losses = []
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc="train"):

        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    total += targets.shape[0]
    correct += torch.sum(outputs.max(1)[1] == targets).item()

    return {
        'train_acc': np.round(correct / total, 3),
        'train_loss': np.round(np.mean(losses), 3),
    }


def validate(model, val_loader, device, criterion, task):
    model.eval()
    model = model.to(device)

    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="validate"):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)

            losses.append(loss.item())
            total += targets.shape[0]
            correct += (outputs.max(1)[1] == targets).sum().cpu().numpy()

        return {
            'val_acc': np.round(correct / total, 3),
            'val_loss': np.round(np.mean(losses), 3),
        }
    
centralized_model = Net(in_channels=ds_info['n_channels'],
                        num_classes=len(ds_info['label']))
optimizer = torch.optim.Adam(centralized_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


val_acc = []
val_loss = []
train_acc = []
train_loss = []

# Start!
history = validate(centralized_model,
                   test_dl,
                   device=DEVICE,
                   criterion=criterion,
                   task=ds_info['task'])
print('Before training: ', history)


for epoch in range(NUM_EPOCHS):
    train_history = train(centralized_model,
                          train_dl,
                          device=DEVICE,
                          optimizer=optimizer,
                          criterion=criterion,
                          task=ds_info['task'])
    
    train_acc.append(train_history['train_acc'])
    train_loss.append(train_history['train_loss'])

    val_history = validate(centralized_model,
                           test_dl,
                           device=DEVICE,
                           criterion=criterion,
                           task=ds_info['task'])
    
    val_acc.append(val_history['val_acc'])
    val_loss.append(val_history['val_loss'])

    print(f'Epoch {epoch}: {train_history} - {val_history}')

    


# Should be the same as defined in `director_config.yaml`
director_node_fqdn = 'localhost'
director_port = 50051

director_workspace_path = 'director'
director_config_file = os.path.join(director_workspace_path,'director_config.yaml')
director_logfile = os.path.join(director_workspace_path, 'director.log')

# Start director
# os.system(f'fx director start --disable-tls -c {director_config_file} '
#           f'>{director_logfile} &')
# !sleep 5 && tail -n5 $director_logfile

def generate_envoy_configs(
        config: dict,
        n_cols: int,
        datapath: str,
        dataname: str,
        save_path: str) -> list:
    
    config_paths = list()
    for i in range(1, n_cols+1):
        path = os.path.abspath(os.path.join(save_path, f'{i}_envoy_config.yaml'))
        config['shard_descriptor']['params']['datapath'] = datapath
        config['shard_descriptor']['params']['dataname'] = dataname    
        config['shard_descriptor']['params']['rank_worldsize'] = f'{i},{n_cols}'
        with open(path, 'w') as f:
            yaml.safe_dump(config, f)
        config_paths.append(path)
    return config_paths

# Read the original envoy config file content
tutorial_dir = '/home/shubham/Academics/Thesis/openfl/openfl-tutorials/interactive_api/PyTorch_MedMNIST_2D'
original_config_path = os.path.join(tutorial_dir, 'envoy', 'envoy_config.yaml')
with open(original_config_path, 'r') as f:
    original_config = yaml.safe_load(f)

# Generate configs for as many envoys
# config_paths = generate_envoy_configs(original_config,
#                                       n_cols=1,
#                                       datapath=DATASET_PATH,
#                                       dataname=DATASET_NAME,
#                                       save_path=os.path.dirname(original_config_path))
# Start envoys in a loop
# cwd = os.getcwd()
# for i, path in enumerate(config_paths):
#     print(f'Starting Envoy {i+1}')
#     os.chdir(os.path.dirname(path))

#     # Wait until envoy loads dataset
#     os.system(f'fx envoy start -n env_{i+1} --disable-tls '
#                 f'--envoy-config-path {path} -dh {director_node_fqdn} -dp {director_port} '
#                 f'>env_{i+1}.log 2>&1 &')
#     !grep -q "MedMNIST data was loaded" <( tail -f env_{i+1}.log )
    
#     os.chdir(cwd)

# Create a federation
from openfl.interface.interactive_api.federation import Federation

# Federation can also determine local fqdn automatically
federation = Federation(client_id='env_1',
                        director_node_fqdn=director_node_fqdn,
                        director_port=director_port,
                        tls=False)

# Data scientist may request a list of connected envoys
shard_registry = federation.get_shard_registry()
pprint(shard_registry)

from openfl.interface.interactive_api.experiment import DataInterface


class MedMnistFedDataset(DataInterface):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor

        self.train_set = TransformDataset(
            self._shard_descriptor.get_dataset('train'), transform=transform)

        self.valid_set = TransformDataset(
            self._shard_descriptor.get_dataset('val'), transform=transform)

    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        return DataLoader(self.train_set,
                          num_workers=8,
                          batch_size=self.kwargs['train_bs'],
                          shuffle=True)

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        return DataLoader(self.valid_set,
                          num_workers=8,
                          batch_size=self.kwargs['valid_bs'])

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_set)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.valid_set)

fed_dataset = MedMnistFedDataset(train_bs=BATCH_SIZE, valid_bs=BATCH_SIZE)

from openfl.interface.interactive_api.experiment import ModelInterface

model = Net(in_channels=ds_info['n_channels'],
            num_classes=len(ds_info['label']))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'

MI = ModelInterface(model=model,
                    optimizer=optimizer,
                    framework_plugin=framework_adapter)

from openfl.interface.interactive_api.experiment import TaskInterface

# Task interface currently supports only standalone functions.
TI = TaskInterface()
criterion = nn.CrossEntropyLoss()
fixed = {'criterion': criterion, 'task': ds_info['task']}

# Train task
TI.add_kwargs(**fixed)(TI.register_fl_task(model='model',
                                           data_loader='train_loader',
                                           device='device',
                                           optimizer='optimizer')(train))

# Validate task
TI.add_kwargs(**fixed)(TI.register_fl_task(model='model',
                                           device='device',
                                           data_loader='val_loader')(validate))

from openfl.interface.interactive_api.experiment import FLExperiment

fl_experiment = FLExperiment(federation=federation,
                             experiment_name='bloodmnist2d_experiment')
fl_experiment.start(model_provider=MI,
                    task_keeper=TI,
                    data_loader=fed_dataset,
                    rounds_to_train=3,
                    device_assignment_policy='CUDA_PREFERRED')

# This method not only prints messages recieved from the director,
# but also saves logs in the tensorboard format (by default)
fed_val_acc = []
fed_val_loss = []
fed_train_acc = []
fed_train_loss = []

metrics_to_plot = fl_experiment.stream_metrics(tensorboard_logs=True)

if metrics_to_plot is not None:
    for metric_output in metrics_to_plot:
        flag, val, round = metric_output
        if flag == "val_acc":
            fed_val_acc.append(val)
        elif flag == "val_loss":
            fed_val_loss.append(val)
        elif flag == "train_acc":
            fed_train_acc.append(val)
        elif flag == "train_loss":
            fed_train_loss.append(val)   

    x = [1,2,3]
    fig, ax = plt.subplots(2,2)

    ax[0,0].plot(x, fed_val_acc, "-r", label="Federated")
    ax[0,0].plot(x, val_acc, "-b", label="Centralized")
    ax[0,0].set_xlabel("Epoch/Round")
    ax[0,0].set_ylabel("Acuracy")
    ax[0,0].set_title("Validation Accuracy")
    ax[0,0].legend()

    ax[0,1].plot(x, fed_val_loss, "-r", label="Federated")
    ax[0,1].plot(x, val_loss, "-b", label="Centralized")
    ax[0,1].set_xlabel("Epoch/Round")
    ax[0,1].set_ylabel("loss")
    ax[0,1].set_title("Validation Loss")
    ax[0,1].legend()

    ax[1,0].plot(x, fed_train_acc, "-r", label="Federated")
    ax[1,0].plot(x, train_acc, "-b", label="Centralized")
    ax[1,0].set_xlabel("Epoch/Round")
    ax[1,0].set_ylabel("Acuracy")
    ax[1,0].set_title("Training Accuracy")
    ax[1,0].legend()

    ax[1,1].plot(x, fed_train_loss, "-r", label="Federated")
    ax[1,1].plot(x, train_loss, "-b", label="Centralized")
    ax[1,1].set_xlabel("Epoch/Round")
    ax[1,1].set_ylabel("loss")
    ax[1,1].set_title("Training Loss")
    ax[1,1].legend()

    plt.show()
    input()