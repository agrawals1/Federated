import json
import os
import numpy as np
import wget
from ...ml.engine import ml_engine_adapter
import zipfile
from ...constants import FEDML_DATA_MNIST_URL
import logging
from scenarios.abstract_scenario import AbstractScenario
import torch
from torch.utils.data import DataLoader, TensorDataset

cwd = os.getcwd()

def load_data(args, train, test, dev):
    # Assuming args has attributes 'client_num' and 'batch_size'
    clients_num = range(args.client_num_in_total)

    # Creating TensorDatasets for train, test, and dev sets
    train_dataset = TensorDataset(train.g, train.w, train.x, train.y, train.z)
    test_dataset = TensorDataset(test.g, test.w, test.x, test.y, test.z)
    dev_dataset = TensorDataset(dev.g, dev.w, dev.x, dev.y, dev.z)

    # Creating DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    # Dictionaries to hold the data for each client
    train_data_local_dict = {user: [] for user in clients_num}
    test_data_local_dict = {user: [] for user in clients_num}
    val_data_local_dict = {user: [] for user in clients_num}
    train_data_local_num_dict = {}
    test_data_local_num_dict = {}
    val_data_local_num_dict = {}

    # Distributing the batches among clients
    for i, (train_batch, test_batch, dev_batch) in enumerate(zip(train_loader, test_loader, dev_loader)):
        user = i % args.client_num_in_total
        train_data_local_dict[user].append(train_batch)
        test_data_local_dict[user].append(test_batch)
        val_data_local_dict[user].append(dev_batch)

    # Calculating the number of samples for each client
    for user in clients_num:
        train_data_local_num_dict[user] = len(train_data_local_dict[user]) * args.batch_size
        test_data_local_num_dict[user] = len(test_data_local_dict[user]) * args.batch_size
        val_data_local_num_dict[user] = len(val_data_local_dict[user]) * args.batch_size

    return (
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        train_data_local_num_dict,
        test_data_local_num_dict,
        val_data_local_num_dict
    )


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
    scenario = AbstractScenario(filename="data/zoo/" + args.scenario_name + ".npz")
    scenario.info()
    scenario.to_tensor()
    scenario.to_cuda()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")
    
    train_data_local_dict, test_data_local_dict,\
    val_data_local_dict, train_data_local_num_dict,\
    test_data_local_num_dict, val_data_local_num_dict = load_data(args, train, test, dev)

    return (
        args.client_num_in_total,
        train.y.shape[0],
        test.y.shape[0],
        dev.y.shape[0],
        train,
        test,
        dev,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        None,
    )
