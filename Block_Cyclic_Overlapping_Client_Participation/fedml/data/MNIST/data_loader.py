import json
import os
import seaborn as sns
import numpy as np
from ...ml.engine import ml_engine_adapter
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from ...constants import FEDML_DATA_MNIST_URL
from torchvision import datasets, transforms
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

current_dir = Path(__file__).resolve()

def distribute_test_data(test_dataset, num_clients) -> Dict[int, List[int]]:
    all_indices = list(range(len(test_dataset)))
    np.random.shuffle(all_indices)
    test_partition = {}
    split_size = len(test_dataset) // num_clients
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i+1) * split_size if i != num_clients-1 else len(test_dataset)
        test_partition[i] = all_indices[start_idx:end_idx]
    return test_partition


def plot_client_data_distribution(args, stats: Dict, num_classes: int) -> None:
    client_num = len(stats) - 1
    ggp_dir = current_dir.parents[2]
    save_path = os.path.join(str(ggp_dir), "ClassAndSampleDistributionAmongClients")
    save_id = str(args.alpha_dirichlet)
    
    fig, axs = plt.subplots(2, figsize=(15, 12))

    unique_classes_counts = [len(stat['y']) for stat in stats.values() if 'y' in stat]
    axs[0].bar(range(client_num), unique_classes_counts, color='skyblue')
    axs[0].set_xlabel('Clients')
    axs[0].set_ylabel('Number of Unique Classes')
    axs[0].set_title('Number of Unique Classes Each Client Possesses')
    axs[0].set_xticks(list(range(0, client_num, 10)))
    axs[0].set_xticklabels([f'Client {i+1}' for i in range(0, client_num, 10)])

    total_samples = [stat['x'] for stat in stats.values() if 'x' in stat]
    axs[1].bar(range(client_num), total_samples, color='skyblue')
    axs[1].set_xlabel('Clients')
    axs[1].set_ylabel('Total Samples')
    axs[1].set_title('Total Samples for Each Client')
    axs[1].set_xticks(list(range(0, client_num, 10)))
    axs[1].set_xticklabels([f'Client {i+1}' for i in range(0, client_num, 10)])

    mean_samples = np.mean(total_samples)
    std_samples = np.std(total_samples)
    axs[1].axhline(y=mean_samples, color='r', linestyle='-', label=f"Mean: {mean_samples:.2f}")
    axs[1].axhline(y=mean_samples + std_samples, color='g', linestyle='--', label=f"Mean + 1 StdDev: {mean_samples + std_samples:.2f}")
    axs[1].axhline(y=mean_samples - std_samples, color='g', linestyle='--', label=f"Mean - 1 StdDev: {mean_samples - std_samples:.2f}")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(save_path + save_id + ".png")
    plt.show()
    
    
def dirichlet_distribution(labels: List[int], client_num: int, alpha: float, least_samples: int = 20, seed: int = None) -> Tuple[Dict[int, List[int]], Dict]:
    if seed is not None:
        np.random.seed(seed)
    label_num = len(set(labels))  # Number of unique labels
    stats = {}
    min_size = 0
    targets_numpy = np.array(labels, dtype=np.int32)
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]

    while min_size < least_samples:
        data_indices = [[] for _ in range(client_num)]
        for k in range(label_num):
            np.random.shuffle(data_idx_for_each_label[k])
            distrib = np.random.dirichlet(np.repeat(alpha, client_num))
            distrib = np.array(
                [
                    p * (len(idx_j) < len(targets_numpy) / client_num)
                    for p, idx_j in zip(distrib, data_indices)
                ]
            )
            distrib = distrib / distrib.sum()
            distrib = (np.cumsum(distrib) * len(data_idx_for_each_label[k])).astype(int)[:-1]
            data_indices = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(
                    data_indices, np.split(data_idx_for_each_label[k], distrib)
                )
            ]
        min_size  = min([len(idx_j) for idx_j in data_indices])

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets_numpy[data_indices[i]])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "mean": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    data_indices = {k:v.tolist() for k,v in enumerate(data_indices)}

    return data_indices, stats

def distribute_classes_among_clients(num_clients: int, total_classes: int) -> List[List[int]]:
    if total_classes >= num_clients:
        classes_per_client = total_classes // num_clients
        remaining_classes = total_classes % num_clients

        client_data_division = [list(range(i * classes_per_client, (i+1) * classes_per_client)) for i in range(num_clients)]

        # Handle the remaining classes
        for idx, cl in enumerate(range(total_classes - remaining_classes, total_classes)):
            client_data_division[idx].append(cl)
    else:
        client_data_division = [[i] for i in range(total_classes)]
        for idx in range(total_classes, num_clients):
            client_data_division.append([idx % total_classes])

    return client_data_division

def process_combined_dataset(train_dataset, test_dataset):
    """
    Combine and process training and testing datasets.
    """
    all_x = [img.numpy().tolist() for img, _ in train_dataset] + [img.numpy().tolist() for img, _ in test_dataset]
    all_y = [label for _, label in train_dataset] + [label for _, label in test_dataset]
    return all_x, all_y

def load_dataset(dataset_name, transform):
    """Load a dataset based on its name and get the number of unique classes.
    
    Parameters:
    - dataset_name (str): Name of the dataset.
    - transform: Image transformations to apply.
    
    Returns:
    - tuple: Train dataset, test dataset, and number of classes.
    """
    
    root_dir = "./home/shubham/fed_data/"
    # Dictionary mapping dataset names to their corresponding classes, directories, and number of classes
    dataset_classes = {
        "MNIST": {
            "class": datasets.MNIST,
            "dir": "MNIST",
            "num_classes": 10
        },
        "FashionMNIST": {
            "class": datasets.FashionMNIST,
            "dir": "FashionMNIST",
            "num_classes": 10
        },
        "CIFAR100": {
            "class": datasets.CIFAR100,
            "dir": "CIFAR100",
            "num_classes": 100
        },
        "CIFAR10": {
            "class": datasets.CIFAR10,
            "dir": "CIFAR10",
            "num_classes": 10
        }
    }
    
    # Get the relevant class, directory, and number of classes
    dataset_info = dataset_classes.get(dataset_name, dataset_classes["CIFAR100"])
    dataset_class = dataset_info["class"]
    dataset_dir = dataset_info["dir"]
    num_classes = dataset_info["num_classes"]
    
    # Load the datasets
    train_dataset = dataset_class(root=os.path.join(root_dir, dataset_dir), train=True, transform=transform, download=True)
    test_dataset = dataset_class(root=os.path.join(root_dir, dataset_dir), train=False, transform=transform, download=True)
    
    return train_dataset, test_dataset, num_classes

def read_data_dirichlet(args, dataset_name, alpha, num_clients=7):
    transform = transforms.Compose([

    transforms.ToTensor(),               # Convert to tensor (required for training)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize the images
])

    
    train_dataset, test_dataset, num_classes = load_dataset(dataset_name, transform)

    all_data_x, all_data_y = process_combined_dataset(train_dataset, test_dataset)
 
    # Splitting data using Dirichlet distribution
    train_client_idcs, stats = dirichlet_distribution(train_dataset, num_clients, alpha, least_samples = 32, seed=args.dirichlet_seed)

    plot_client_data_distribution(args, stats, num_classes)
    
    # Construct data dictionaries
    train_data = {}
    test_data = {}
    for client_idx, idcs in client_idcs.items():
        client_data_x = [all_data_x[i] for i in idcs]
        client_data_y = [all_data_y[i] for i in idcs]

        # Split data into train and test sets for each client
        split_idx = int(len(client_data_x) * 0.8)  # Assuming an 80-20 train-test split
        train_data[client_idx] = {"x": client_data_x[:split_idx], "y": client_data_y[:split_idx]}
        test_data[client_idx] = {"x": client_data_x[split_idx:], "y": client_data_y[split_idx:]}

    clients = list(range(num_clients))
    groups = []

    return clients, groups, train_data, test_data, num_classes

# Note: We have provided placeholders for "news" dataset's lengths. We will need additional information to handle them properly.


def read_data_hetero(train_data_dir, test_data_dir, num_clients=7):
    """
    Reads and processes heterogeneous training and testing data for a given number of clients.

    Parameters:
    - train_data_dir (str): Path to the directory containing training data files. Each file is assumed to be a .json.
    - test_data_dir (str): Path to the directory containing testing data files. Each file is assumed to be a .json.
    - num_clients (int, optional): Number of clients for which the data should be processed. Default is 7.

    Returns:
    - clients (list of int): List of client indices.
    - groups (list): Currently, an empty list is returned. Can be expanded for further use if needed.
    - train_data (dict): Dictionary where keys are client indices and values are dictionaries with 'x' and 'y' 
                         representing training data features and labels respectively for each client.
    - test_data (dict): Dictionary where keys are client indices and values are dictionaries with 'x' and 'y' 
                        representing testing data features and labels respectively for each client.
    """
    # Load all data from JSON files
    all_train_data = {}
    all_test_data = {}

    train_files = [f for f in os.listdir(train_data_dir) if f.endswith(".json")]
    for f in train_files:
        with open(os.path.join(train_data_dir, f), "r") as inf:
            cdata = json.load(inf)
        all_train_data.update(cdata["user_data"])

    test_files = [f for f in os.listdir(test_data_dir) if f.endswith(".json")]
    for f in test_files:
        with open(os.path.join(test_data_dir, f), "r") as inf:
            cdata = json.load(inf)
        all_test_data.update(cdata["user_data"])

    # Create a mapping from class label to data
    total_classes = 10
    class_to_data_train = {i: [] for i in range(total_classes)}
    class_to_data_test = {i: [] for i in range(total_classes)}

    for _, data in all_train_data.items():
        for x, y in zip(data["x"], data["y"]):
            class_to_data_train[y].append((x, y))
    
    for _, data in all_test_data.items():
        for x, y in zip(data["x"], data["y"]):
            class_to_data_test[y].append((x, y))

    # Distribute data among clients based on class labels
    client_data_division = distribute_classes_among_clients(num_clients, total_classes)

    train_data = {}
    test_data = {}

    for client_idx, classes in enumerate(client_data_division):
        train_data[client_idx] = {
            "x": [x for c in classes for x, _ in class_to_data_train[c]],
            "y": [y for c in classes for _, y in class_to_data_train[c]]
        }
        test_data[client_idx] = {
            "x": [x for c in classes for x, _ in class_to_data_test[c]],
            "y": [y for c in classes for _, y in class_to_data_test[c]]
        }

    clients = list(range(num_clients))
    groups = []

    return clients, groups, train_data, test_data


def batch_data(args, data, batch_size):
    
    min_batch_size = 20
    """
    Splits data into batches of a specified size.

    Parameters:
    - args: Arguments containing various configuration parameters. 
            Specifically, this function checks for a 'model' attribute.
    - data (dict): Data to be batched. It should contain 'x' and 'y' keys.
    - batch_size (int): Desired batch size.

    Returns:
    - list: List of batches where each batch is a tuple (batched_x, batched_y).
    """
    assert 2 <= min_batch_size < batch_size # min_batch_size should be between 2 and batch_size
    
    data_x = data["x"]
    data_y = data["y"]
    
    # Shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # Calculate total number of batches
    n_batches = len(data_x) // batch_size
    remainder = len(data_x) % batch_size
    
    # If the remainder is less than min_batch_size, n_batches-=1
    if remainder >= min_batch_size:
        n_batches += 1
    
    # Create batches
    batched_data = []
    for i in range(n_batches):
        start_idx = i * batch_size
        if i == n_batches - 1:
            if remainder >= min_batch_size:
                end_idx = start_idx + remainder
            else:
                end_idx = start_idx + batch_size + remainder
        else:
            end_idx = start_idx + batch_size
            
        batched_x = data_x[start_idx: end_idx]
        batched_y = data_y[start_idx: end_idx]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batched_data.append((batched_x, batched_y))
    
    return batched_data



def load_partition_data_by_device_id(args, device_id, base_train_path="MNIST_mobile", base_test_path="MNIST_mobile"):
    """Generates paths based on a device ID and then loads partitioned MNIST data.

    Parameters:
    - args: Arguments containing various configuration parameters.
    - device_id (str): Device identifier.
    - base_train_path (str, optional): Base path to the training data. Defaults to "MNIST_mobile".
    - base_test_path (str, optional): Base path to the testing data. Defaults to "MNIST_mobile".

    Returns:
    - tuple: Tuple containing loaded data and related information.
    """
    train_path = os.path.join(base_train_path, device_id, "train")
    test_path = os.path.join(base_test_path, device_id, "test")
    return load_partition_data(args, train_path, test_path)



def load_partition_data(args):
    """Loads and partitions data for MNIST using the read_data_dirichlet function.

    Parameters:
    - args: Arguments containing various configuration parameters.

    Returns:
    - tuple: Tuple containing various data-related metrics and structures.
    """
    # Load and partition data
    users, groups, train_data, test_data, num_classes = read_data_dirichlet(args, args.dataset, args.alpha_dirichlet, args.client_num_in_total)

    train_data_num = sum([len(data["x"]) for data in train_data.values()])
    test_data_num = sum([len(data["x"]) for data in test_data.values()])
    
    # Convert to batches
    train_data_local_dict = {client: batch_data(args, data, args.batch_size) for client, data in train_data.items()}
    test_data_local_dict = {client: batch_data(args, data, args.batch_size) for client, data in test_data.items()}

    # Global data
    train_data_global = [batch for batches in train_data_local_dict.values() for batch in batches]
    test_data_global = [batch for batches in test_data_local_dict.values() for batch in batches]

    # Data stats
    train_data_local_num_dict = {client: len(data["x"]) for client, data in train_data.items()}
    client_num = len(users)
    class_num = num_classes 

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num
    )

