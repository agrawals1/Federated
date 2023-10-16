import json
import os
import seaborn as sns
import numpy as np
import wget
from ...ml.engine import ml_engine_adapter
import matplotlib.pyplot as plt

cwd = os.getcwd()

import zipfile

from ...constants import FEDML_DATA_MNIST_URL
import logging
import torch
from torchvision import datasets, transforms
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

def distribute_test_data(test_dataset, num_clients):
    all_indices = list(range(len(test_dataset)))
    np.random.shuffle(all_indices)
    test_partition = {}
    split_size = len(test_dataset) // num_clients
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i+1) * split_size if i != num_clients-1 else len(test_dataset)
        test_partition[i] = all_indices[start_idx:end_idx]
    return test_partition

# def plots(stats, num_of_clients):
#     print("*********** Some stats on client data distribution **********")
#     print(f"Mean samples per client: {stats['sample per client']['std']:.2f}")
#     print()
#     print(f"Standard deviation of samples per client: {stats['sample per client']['stddev']:.2f}")
#     num_of_clients = min(10, num_of_clients)
#     samples_per_client = [v["x"] for k, v in stats.items() if k != "sample per client"]
#     client_indices = [i for i, _ in enumerate(samples_per_client)]
    
#     plt.figure(figsize=(10,6))
    
#     # If there are 10 or fewer clients
#     if len(samples_per_client) <= 10:
#         plt.bar(client_indices, samples_per_client, color="skyblue", edgecolor="black")
#     # If there are more than 10 clients
#     else:
#         sorted_samples_with_indices = sorted(enumerate(samples_per_client), key=lambda x: x[1])
#         top_5_indices, top_5_samples = zip(*sorted_samples_with_indices[-5:])
#         bottom_5_indices, bottom_5_samples = zip(*sorted_samples_with_indices[:5])
        
#         plt.bar(top_5_indices, top_5_samples, color="green", label="Top 5 clients (most samples)", edgecolor="black")
#         plt.bar(bottom_5_indices, bottom_5_samples, color="red", label="Bottom 5 clients (least samples)", edgecolor="black")
#         plt.legend()

#     plt.title("Frequency plot of samples per client")
#     plt.xlabel("Client Index")
#     plt.ylabel("Number of samples")
#     plt.grid(True, which='both', linestyle="--", linewidth=0.5)
#     plt.savefig("/home/shubham/Federated/Block_Cyclic_Overlapping_Client_Participation/data/samples_per_client_histogram.png")  # Save the plot
#     plt.close()  # Close the plot

#     plt.figure(figsize=(12,6))
#     for i in range(num_of_clients):
#         class_count = stats[i]["y"]
#         labels = list(class_count.keys())
#         counts = list(class_count.values())
#         plt.subplot(1, num_of_clients, i+1)
#         plt.bar(labels, counts, color="skyblue", edgecolor="black")
#         plt.title(f"Client {i} Class Distribution")
#         plt.xlabel("Class Label")
#         plt.ylabel("Number of Samples")
#         plt.grid(True, which="both", linestyle="--", linewidth="0.5")
#     plt.tight_layout()
#     plt.savefig("/home/shubham/Federated/Block_Cyclic_Overlapping_Client_Participation/data/class_distribution_per_client.png")  # Save the plot
#     plt.close()  # Close the plot

def plots(args, stats, num_classes):
    
    client_num = len(stats) - 1
    save_path = "/home/shubham/Federated/Block_Cyclic_Overlapping_Client_Participation/data/"
    save_id = args.run_name
    
    fig, axs = plt.subplots(2, figsize=(15, 12))

    # Plot - Number of unique classes possessed by each client
    unique_classes_counts = [len(stat['y']) for stat in stats.values() if 'y' in stat]
    axs[0].bar(range(client_num), unique_classes_counts, color='skyblue')
    axs[0].set_xlabel('Clients')
    axs[0].set_ylabel('Number of Unique Classes')
    axs[0].set_title('Number of Unique Classes Each Client Possesses')
    axs[0].set_xticks(list(range(0, client_num, 10)))
    axs[0].set_xticklabels([f'Client {i+1}' for i in range(0, client_num, 10)])

    # Plot - Total samples for each client
    total_samples = [stat['x'] for stat in stats.values() if 'x' in stat]
    axs[1].bar(range(client_num), total_samples, color='skyblue')
    axs[1].set_xlabel('Clients')
    axs[1].set_ylabel('Total Samples')
    axs[1].set_title('Total Samples for Each Client')
    axs[1].set_xticks(list(range(0, client_num, 10)))
    axs[1].set_xticklabels([f'Client {i+1}' for i in range(0, client_num, 10)])

    # Mean and standard deviation lines for total samples
    mean_samples = np.mean(total_samples)
    std_samples = np.std(total_samples)
    axs[1].axhline(y=mean_samples, color='r', linestyle='-', label=f"Mean: {mean_samples:.2f}")
    axs[1].axhline(y=mean_samples + std_samples, color='g', linestyle='--', label=f"Mean + 1 StdDev: {mean_samples + std_samples:.2f}")
    axs[1].axhline(y=mean_samples - std_samples, color='g', linestyle='--', label=f"Mean - 1 StdDev: {mean_samples - std_samples:.2f}")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(save_path + save_id + ".png")
    plt.show()
    
    
def dirichlet(
    dataset: Dataset, client_num: int, alpha: float, least_samples: int
) -> Tuple[Dict, Dict]:
    label_num = len(dataset.classes)
    min_size = 0
    stats = {}

    targets_numpy = np.array(dataset.targets, dtype=np.int32)
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
            distrib = (np.cumsum(distrib) * len(data_idx_for_each_label[k])).astype(
                int
            )[:-1]
            data_indices = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(
                    data_indices, np.split(data_idx_for_each_label[k], distrib)
                )
            ]
            min_size = min([len(idx_j) for idx_j in data_indices])

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets_numpy[data_indices[i]])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    data_indices = {k:v.tolist() for k,v in enumerate(data_indices)}

    return data_indices, stats

def download_mnist(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir, exist_ok=True)

    file_path = os.path.join(data_cache_dir, "MNIST.zip")
    logging.info(file_path)

    # Download the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(FEDML_DATA_MNIST_URL, out=file_path)

    file_extracted_path = os.path.join(data_cache_dir, "MNIST")
    if not os.path.exists(file_extracted_path):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_cache_dir)

def distribute_classes(num_clients, total_classes):
    """
    Distribute a given number of classes among a specified number of clients.

    Parameters:
    - num_clients (int): The number of clients among which the classes should be distributed.
    - total_classes (int): The total number of available classes.

    Returns:
    - list of list of int: A list where each inner list represents the class indices allocated to a client.

    Notes:
    - If total_classes is greater than or equal to num_clients, the function will try to allocate an equal
      number of classes to each client. Any remaining classes will be distributed one by one starting from the first client.
    - If total_classes is less than num_clients, each client will get at least one class. The clients beyond 
      the total_classes will get classes in a repeated manner.
    """
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

# def split_noniid(train_idcs, train_labels, alpha, n_clients):
#     '''
#     Splits a list of data indices with corresponding labels
#     into subsets according to a dirichlet distribution with parameter
#     alpha
#     '''
#     n_classes = train_labels.max()+1    
#     alpha_list = [float(alpha) for _ in range(n_clients)]
#     label_distribution = np.random.dirichlet(alpha_list, n_classes)

#     class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
#            for y in range(n_classes)]

#     client_idcs = [[] for _ in range(n_clients)]
#     for c, fracs in zip(class_idcs, label_distribution):
#         for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
#             client_idcs[i] += [idcs]

#     client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
  
#     return client_idcs



def read_data_dirichlet(args, alpha, pytorch_dataset, num_clients=7):
    """
    Reads and processes heterogeneous training and testing data for a given number of clients using Dirichlet distribution.

    Parameters:
    - train_data_dir (str): Path to the directory containing training data files. Each file is assumed to be a .json.
    - test_data_dir (str): Path to the directory containing testing data files. Each file is assumed to be a .json.
    - alpha: A parameter for the Dirichlet distribution.
    - num_clients (int, optional): Number of clients for which the data should be processed. Default is 7.

    Returns:
    - clients (list of int): List of client indices.
    - groups (list): Currently, an empty list is returned. Can be expanded for further use if needed.
    - train_data (dict): Dictionary where keys are client indices and values are dictionaries with 'x' and 'y' 
                         representing training data features and labels respectively for each client.
    - test_data (dict): Dictionary where keys are client indices and values are dictionaries with 'x' and 'y' 
                        representing testing data features and labels respectively for each client.

    Notes:
    - The function assumes that the training and testing data are saved in .json format and each file's content has a 
      key named "user_data".
    """
    
    transform = transforms.Compose([transforms.ToTensor()])
    if pytorch_dataset == "mnist":
        train_dataset = datasets.MNIST(root="./home/shubham/fed_data/MNIST", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="./home/shubham/fed_data/MNIST", train=False, transform=transform, download=True)
        all_train_x = [img.flatten().numpy().tolist() for img,_ in train_dataset]
        all_test_x = [img.flatten().numpy().tolist() for img,_ in test_dataset]
        all_train_y = [label for _, label in train_dataset]
        all_test_y = [label for _, label in test_dataset]
    elif pytorch_dataset == "fashionMnist" :
        train_dataset = datasets.FashionMNIST(root="./home/shubham/fed_data/Fashion", train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root="./home/shubham/fed_data/Fashion", train=False, transform=transform, download=True)
    elif pytorch_dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root="./home/shubham/fed_data/CIFAR100", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR100(root="./home/shubham/fed_data/CIFAR100", train=False, transform=transform, download=True)
        all_train_x = [img.numpy().tolist() for img,_ in train_dataset]
        all_test_x = [img.numpy().tolist() for img,_ in test_dataset]
        all_train_y = [label for _, label in train_dataset]
        all_test_y = [label for _, label in test_dataset]
    else:
        train_dataset = datasets.CIFAR10(root="./home/shubham/fed_data/CIFAR", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root="./home/shubham/fed_data/CIFAR", train=False, transform=transform, download=True)
        all_train_x = [img.numpy().tolist() for img,_ in train_dataset]
        all_test_x = [img.numpy().tolist() for img,_ in test_dataset]
        all_train_y = [label for _, label in train_dataset]
        all_test_y = [label for _, label in test_dataset]
    # Splitting data using Dirichlet distribution
    train_client_idcs, stats = dirichlet(train_dataset, num_clients, alpha, len(all_train_y)//(5*num_clients))
    plots(args, stats, 100)
    # test_client_idcs = dirichlet(np.arange(len(all_test_y)), np.array(all_test_y), alpha, num_clients)
    # uniform distribution of test data
    test_client_idcs = distribute_test_data(test_dataset, num_clients)
    
    train_data = {}
    test_data = {}
    for client_idx, idcs in train_client_idcs.items():
        client_data_x = []
        client_data_y = []
    
        for i in idcs:
            client_data_x.append(all_train_x[i])
            client_data_y.append(all_train_y[i])
    
        train_data[client_idx] = {"x": client_data_x, "y": client_data_y}

    for client_idx, idcs in test_client_idcs.items():
        client_data_x = []
        client_data_y = []
    
        for i in idcs:
            client_data_x.append(all_test_x[i])
            client_data_y.append(all_test_y[i])
    
        test_data[client_idx] = {"x": client_data_x, "y": client_data_y}

    clients = list(range(num_clients))
    groups = []

    return clients, groups, train_data, test_data


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

    Notes:
    - The function assumes that the training and testing data are saved in .json format and each file's content has a 
      key named "user_data".
    - Data is read and then mapped based on class labels to distribute it heterogeneously among clients using the 
      distribute_classes function.
    """
    all_train_data = {}
    all_test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        all_train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
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

    client_data_division = distribute_classes(num_clients, total_classes)

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






def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    clients = sorted(cdata["users"])

    return clients, groups, train_data, test_data


def batch_data(args, data, batch_size):

    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size, device_id, train_path="MNIST_mobile", test_path="MNIST_mobile"):
    train_path += os.path.join("/", device_id, "train")
    test_path += os.path.join("/", device_id, "test")
    return load_partition_data_mnist(batch_size, train_path, test_path)


def load_partition_data_mnist(
    args, batch_size
):
    users, groups, train_data, test_data = read_data_dirichlet(args, args.alpha_dirichlet,num_clients=args.client_num_in_total, pytorch_dataset = args.dataset)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[u]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(args, train_data[u], batch_size)
        test_batch = batch_data(args, test_data[u], batch_size)
        
        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
