import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random

class AGMMZooModified:
    def __init__(self, g_function='linear', two_gps=False, n_instruments=1, iv_strength=0.5, z_range=(-3, 3), confounder_variance=1.0):
        self._function_name = g_function
        self._two_gps = two_gps
        self._n_instruments = n_instruments
        self._iv_strength = iv_strength
        self.z_range = z_range
        self.confounder_variance = confounder_variance
    
    def generate_data(self, num_data):
        confounder = np.random.normal(0, self.confounder_variance, size=(num_data, 1))
        z = np.random.uniform(self.z_range[0], self.z_range[1], size=(num_data, self._n_instruments))
        iv_strength = self._iv_strength
        if self._two_gps:
            x = 2 * z[:, 0].reshape(-1, 1) * (z[:, 0] > 0).reshape(-1, 1) * iv_strength \
                + 2 * z[:, 1].reshape(-1, 1) * (z[:, 1] < 0).reshape(-1, 1) * iv_strength \
                + 2 * confounder * (1 - iv_strength) + \
                np.random.normal(0, .1, size=(num_data, 1))
        else:
            x = 2 * z[:, 0].reshape(-1, 1) * iv_strength \
                + 2 * confounder * (1 - iv_strength) + \
                np.random.normal(0, .1, size=(num_data, 1))
        g = self._true_g_function_np(x)
        y = g + 2 * confounder + np.random.normal(0, .1, size=(num_data, 1))
        return torch.tensor(x, dtype=torch.double), torch.tensor(z, dtype=torch.double), torch.tensor(y, dtype=torch.double), torch.tensor(g, dtype=torch.double), torch.tensor(x, dtype=torch.double)

    def _true_g_function_np(self, x):
        if self._function_name == 'abs':
            return np.abs(x)
        elif self._function_name == 'sin':
            return np.sin(x)
        elif self._function_name == 'step':
            return 1. * (x < 0) + 2.5 * (x >= 0)
        elif self._function_name == 'linear':
            return x
        else:
            raise NotImplementedError(f"Function {self._function_name} not implemented")

def preload_mnist(train):
    if train:
        mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    else:
        mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    # Shuffle indices to randomize the data access
    indices = np.arange(len(mnist))
    np.random.shuffle(indices)
    return mnist, indices

def get_mnist_images(mnist, indices, digit_indices):
    # Fetch images based on shuffled indices and transformed digit classes
    return torch.stack([mnist[indices[int(idx)]][0] for idx in digit_indices]).double()

def normalize(y, mean, std):
    # mean = mean.detach().numpy()
    # std = std.detach().numpy()
    return (y - mean) / std

def load_data_modified_v2(client_config, args, num_data_per_client, g_function='linear', batch_size=128, train_split=0.33, test_split=0.33):
    train_data_local_dict = {}
    test_data_local_dict = {}
    val_data_local_dict = {}
    train_data_local_num_dict = {}
    test_data_local_num_dict = {}
    val_data_local_num_dict = {}

    mnist, shuffled_indices = preload_mnist(True)
    mnist_test, shuffled_indices_test = preload_mnist(False)
    π = lambda x: np.clip(1.5 * x + 5.0, 0, 9).round()
    
    num_data_per_client_test = 100
    for client_id, config in client_config.items():
        z_range = config.get('z_range', (-10, 10))
        variance = config.get('variance', 5.0)
        model = AGMMZooModified(g_function=g_function, z_range=z_range, confounder_variance=variance)
        x, z, y, g, w = model.generate_data(num_data_per_client)
        x_t, z_t, y_t, g_t, w_t = model.generate_data(num_data_per_client_test)
        if args.dataset == 'MNIST_GMM':
            x_indices = π(x.numpy().squeeze()).astype(int)
            z_indices = π(z.numpy().squeeze()).astype(int)
            x_indices_t = π(x_t.numpy().squeeze()).astype(int)
            z_indices_t = π(z_t.numpy().squeeze()).astype(int)
            
            x = get_mnist_images(mnist, shuffled_indices, x_indices)
            z = get_mnist_images(mnist, shuffled_indices, z_indices)
            x_t = get_mnist_images(mnist_test, shuffled_indices_test, x_indices_t)
            z_t = get_mnist_images(mnist_test, shuffled_indices_test, z_indices_t)

            # Recalculate g based on the high-dimensional transformation
            g = torch.from_numpy(model._true_g_function_np((x_indices - 5.0) / 1.5).reshape(-1, 1)).double()
            w = torch.from_numpy(x_indices.reshape(-1, 1)).double()  # Update w as per high-dimensional data
            g_t = torch.from_numpy(model._true_g_function_np((x_indices_t - 5.0) / 1.5).reshape(-1, 1)).double()
            w_t = torch.from_numpy(x_indices_t.reshape(-1, 1)).double()  # Update w as per high-dimensional data
        y_mean = y.mean()
        y_std = y.std()
        y = normalize(y, y_mean, y_std)
        g = normalize(g, y_mean, y_std)
        y_t = normalize(y_t, y_mean, y_std)
        g_t = normalize(g_t, y_mean, y_std)
        # Split data into train, test, val
        train_end = int(args.data_per_client * 0.8)
        # test_end = 2 * train_end
        
        # Create TensorDatasets
        train_dataset = TensorDataset(g[:train_end], w[:train_end], x[:train_end], y[:train_end], z[:train_end])
        test_dataset = TensorDataset(g_t, w_t, x_t, y_t, z_t)
        val_dataset = TensorDataset(g[train_end:], w[train_end:], x[train_end:], y[train_end:], z[train_end:])
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        # Extract batches and store them in lists
        train_batches = list(train_loader)
        test_batches = list(test_loader)
        val_batches = list(val_loader)
        
        train_data_local_dict[client_id] = train_batches
        test_data_local_dict[client_id] = test_batches
        val_data_local_dict[client_id] = val_batches
        
        # Counting the number of samples
        
        train_data_local_num_dict[client_id] = batch_size * (len(train_batches) - 1) + len(train_batches[-1][0])
        test_data_local_num_dict[client_id] = batch_size * (len(test_batches) - 1) + len(test_batches[-1][0])
        val_data_local_num_dict[client_id] = batch_size * (len(val_batches) - 1) + len(val_batches[-1][0])


    return (
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        train_data_local_num_dict,
        test_data_local_num_dict,
        val_data_local_num_dict
    )




def concatenate_global_test_data(test_data_local_dict):
    # Initialize lists to hold data tensors from all clients
    g_list = []
    x_list = []
    y_list = []
    z_list = []
    w_list = []

    # Iterate through each client's list of batches
    for client_id, batches in test_data_local_dict.items():
        for batch in batches:
            g, w, x, y, z = batch
            g_list.append(g)
            w_list.append(w)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
    
    # Concatenate all lists to form global tensors
    g_global = torch.cat(g_list, dim=0)
    w_global = torch.cat(w_list, dim=0)
    x_global = torch.cat(x_list, dim=0)
    y_global = torch.cat(y_list, dim=0)
    z_global = torch.cat(z_list, dim=0)

    # Create the global dictionary
    global_test_data = {
        'g': g_global,
        'w': w_global,
        'x': x_global,
        'y': y_global,
        'z': z_global
    }
    return global_test_data

def generate_config(num_clients, hetero, var, z_range):
    client_config = {}
    if hetero:
        # Generate z_range values
        z_range_values = np.linspace(1, 10, num_clients)

        # Generate variance values
        variance_values = np.linspace(1, 5, num_clients)

        # Shuffle variance to pair randomly with z_range
        random.shuffle(variance_values)
        
    else:        
        z_range_values = np.full(num_clients, z_range)
        variance_values = np.full(num_clients, var)
        
    # Create the configuration dictionary
    for client_id in range(num_clients):
        client_config[client_id] = {
            'z_range': (-z_range_values[client_id], z_range_values[client_id]),
            'variance': variance_values[client_id]
        }

        

    return client_config

def load_partition_data_zoo(
    args, batch_size
):
    client_config_example = generate_config(args.client_num_in_total, args.data_hetero, args.homo_zrange, args.homo_var)

    train_data_local_dict, test_data_local_dict,\
        val_data_local_dict, train_data_local_num_dict,\
        test_data_local_num_dict, val_data_local_num_dict  = load_data_modified_v2(client_config_example, args, args.data_per_client, g_function=args.scenario_name, batch_size=args.batch_size)

    test = concatenate_global_test_data(test_data_local_dict)
    train = concatenate_global_test_data(train_data_local_dict)
    val = concatenate_global_test_data(val_data_local_dict)
    return (
            args.client_num_in_total,
            None,
            None,
            None,
            train,
            test,
            val,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,
            None,
        )