import logging
import numpy as np
import random

class ClientSampler:
    def __init__(self, args):
        self.args = args

    def _client_sampling(self, round_idx, client_num_in_total):
        if client_num_in_total == self.args.client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(self.args.client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    
    def client_sampling_cyclic_overlap_pattern(self, round_idx, client_num_in_total):
        start_idx = (round_idx * (self.args.client_num_per_round - self.args.overlap_num)) % client_num_in_total
        client_indexes = [(start_idx + i) % client_num_in_total for i in range(self.args.client_num_per_round)]
        return client_indexes
    
    def client_sampling_cyclic_noOverlap_random(self, round_idx, client_num_in_total):
    # Number of client groups
        K = self.args.num_groups

        # Number of clients in each group
        M_per_K = client_num_in_total // K

        # Determine the active group in the current round
        active_group_idx = round_idx % K

        # Start and end indices for the active group
        start_idx = active_group_idx * M_per_K
        end_idx = start_idx + M_per_K

        # List of clients in the active group
        group_clients = list(range(start_idx, end_idx))

        # Randomly sample N clients from the active group without replacement
        sampled_clients = random.sample(group_clients, self.args.client_num_per_round)

        return sampled_clients
    def client_sampling_cyclic_overlap_random(self, round_idx, client_num_in_total):
    # Number of client groups
        K = self.args.num_groups

        # Percentage overlap P
        P = self.args.overlap_percentage  # Assuming this is given as a value between 0 and 1

        # Number of clients in each group
        M_per_K = client_num_in_total // K

        # Calculate overlap in terms of number of clients
        overlap = int(P * M_per_K)

        # Determine the active group in the current round
        active_group_idx = round_idx % K

        # Start and end indices for the active group
        # Adjust the start index by subtracting the overlap
        start_idx = active_group_idx * M_per_K - overlap * (active_group_idx)
        end_idx = start_idx + M_per_K

        # Adjust for potential negative indices for the first group
        start_idx = max(0, start_idx)

        # List of clients in the active group
        group_clients = list(range(start_idx, end_idx))

        # Randomly sample N clients from the active group without replacement
        sampled_clients = random.sample(group_clients, self.args.client_num_per_round)

        return sampled_clients

