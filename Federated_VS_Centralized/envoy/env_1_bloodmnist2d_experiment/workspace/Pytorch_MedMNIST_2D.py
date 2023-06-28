# Install dependencies if not already installed
import tqdm
import numpy as np
import torch
from pprint import pprint
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
# import subprocess
# output = subprocess.check_output("wandb login", shell=True)
# print(output.decode())
import medmnist

from medmnist import INFO, Evaluator

## Change dataflag here to reflect the ones defined in the envoy_conifg_xxx.yaml
dataname = 'bloodmnist'

# Create a federation
from openfl.interface.interactive_api.federation import Federation

# please use the same identificator that was used in signed certificate
client_id = 'api'
director_node_fqdn = 'localhost'
director_port=50051

# 2) Run with TLS disabled (trusted environment)
# Federation can also determine local fqdn automatically
federation = Federation(
    client_id=client_id,
    director_node_fqdn=director_node_fqdn,
    director_port=director_port, 
    tls=False
)

shard_registry = federation.get_shard_registry()
pprint(shard_registry)

# First, request a dummy_shard_desc that holds information about the federated dataset 
dummy_shard_desc = federation.get_dummy_shard_descriptor(size=10)
dummy_shard_dataset = dummy_shard_desc.get_dataset('train')
sample, target = dummy_shard_dataset[0]
f"Sample shape: {sample.shape}, target shape: {target.shape}"

from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment

num_epochs = 10
TRAIN_BS, VALID_BS = 64, 128

lr = 0.001
gamma=0.1
milestones = [0.5 * num_epochs, 0.75 * num_epochs]

info = INFO[dataname]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

## Data transformations
data_transform = T.Compose([T.ToTensor(), 
                            T.Normalize(mean=[.5], std=[.5])]
                 )

from PIL import Image

class TransformedDataset(Dataset):
    """Image Person ReID Dataset."""


    def __init__(self, dataset, transform=None, target_transform=None):
        """Initialize Dataset."""
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Length of dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
                
        img, label = self.dataset[index]
        
        if self.target_transform:
            label = self.target_transform(label)  
        else:
            label = label.astype(int)
        
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            base_transform = T.PILToTensor()
            img = Image.fromarray(img)
            img = base_transform(img)  

        return img, label

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

        self.train_set = TransformedDataset(
            self._shard_descriptor.get_dataset('train'),
            transform=data_transform
        )       
        
        self.valid_set = TransformedDataset(
            self._shard_descriptor.get_dataset('val'),
            transform=data_transform
        )
        
    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        return DataLoader(
            self.train_set, num_workers=8, batch_size=self.kwargs['train_bs'], shuffle=True)

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        return DataLoader(self.valid_set, num_workers=8, batch_size=self.kwargs['valid_bs'])

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
    
fed_dataset = MedMnistFedDataset(train_bs=TRAIN_BS, valid_bs=VALID_BS)

fed_dataset.shard_descriptor = dummy_shard_desc
for i, (sample, target) in enumerate(fed_dataset.get_train_loader()):
    print(sample.shape, target.shape)

    # define a simple CNN model
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

model = Net(in_channels=n_channels, num_classes=n_classes)
    
# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(model)

from copy import deepcopy

framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=model, optimizer=optimizer, framework_plugin=framework_adapter)

# Save the initial model state
initial_model = deepcopy(model)

TI = TaskInterface()

train_custom_params={'criterion':criterion,'task':task}

# Task interface currently supports only standalone functions.
@TI.add_kwargs(**train_custom_params)
@TI.register_fl_task(model='model', data_loader='train_loader',
                     device='device', optimizer='optimizer')
def train(model, train_loader, device, optimizer, criterion, task):
    total_loss = []
    
    train_loader = tqdm.tqdm(train_loader, desc="train")
    model.train()
    model = model.to(device)
    
    for inputs, targets in tqdm(train_loader, desc="train"):
    
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
            
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            
        total_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()

    return {'train_loss': np.mean(total_loss),}



val_custom_params={'criterion':criterion, 
                   'task':task}

@TI.add_kwargs(**val_custom_params)
@TI.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate(model, val_loader, device, criterion, task):

    val_loader = tqdm.tqdm(val_loader, desc="validate")
    model.eval()
    model.to(device)

    val_score = 0
    total_samples = 0
    total_loss = []
    y_score = torch.tensor([]).to(device)

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
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            
            total_samples += targets.shape[0]
            pred = outputs.argmax(dim=1)
            val_score += pred.eq(targets).sum().cpu().numpy()
        
        acc = val_score / total_samples        
        test_loss = sum(total_loss) / len(total_loss)

        return {'acc': acc,
                'test_loss': test_loss,
                }
    
    # create an experimnet in federation
experiment_name = 'medmnist_exp'
fl_experiment = FLExperiment(federation=federation, experiment_name=experiment_name)

# The following command zips the workspace and python requirements to be transfered to collaborator nodes
fl_experiment.start(model_provider=MI, 
                    task_keeper=TI,
                    data_loader=fed_dataset,
                    rounds_to_train=12,
                    opt_treatment='RESET',
                    device_assignment_policy='CUDA_PREFERRED') 

fl_experiment.stream_metrics(tensorboard_logs=True)
