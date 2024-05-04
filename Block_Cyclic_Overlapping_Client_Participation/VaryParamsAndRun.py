import yaml
import subprocess
from multiprocessing import Process
import pynvml
import os
import argparse 
from multiprocessing import Semaphore
import torch
import time

current_file = os.path.abspath(__file__)
curr_dir = os.path.dirname(current_file)
os.chdir(curr_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
datasets = ["CIFAR100"]
batch_sizes = [32]
strategies = ["client_sampling_cyclic_overlap_pattern"]
learning_rates = [0.2] 
decays = [1e-5]
# betas = [1.0, 100.0, 0.5, 0.1]  # parameter for dirichlet distribution
# betas = [0.1]
# overlaps = [0,3,6]
overlaps = [9] # Client overlap count
comm_rounds = [500]
# seeds = [3087732978, 918854724, 2152041540, 548193746, 993522575, 1531166731, 3136455588, 3525945833, 2018934764, 1770634816]
seeds = [993522575]
group_norms = [0]
AdaptiveDecays = [2]
lr_update_freqs = [5]
data_split = [True]
overlap_types = ["MoreGroups"]


def run_federation_with_semaphore(semaphore, beta, overlap, gpu_id, ot, split, round, lr, bs, decay, seed=6967677, group_norm=0, freq=4, decay_fact=4, dataset="CIFAR10"):
    epochs = 2 if round == 500 else (4 if round == 250 else 1)
    if ot == "MoreClients":
        Remark = "MoreClientsConstantGroups"
    elif ot == "MoreGroups":
        Remark = "MoreGroupsConstantClients"
    else:
        if split:
            Remark = "Data_Split"
        else:
            Remark = "Complete_Data"
    # Remark += "_Data_Split" if split else "_Full_Data"
    run_name = f"Dir:{beta}_Overlap:{overlap}_Remark:{Remark}_FedAvg"
    try:
        run_federation(beta, overlap, gpu_id, ot, round, lr, bs, decay, split, seed, group_norm, freq, decay_fact, epochs, run_name, dataset)
    finally:
        # Release the semaphore when the process is done
        if os.path.exists(os.path.join(curr_dir, f'config/{run_name}.yaml')):
            os.remove(os.path.join(curr_dir, f'config/{run_name}.yaml'))
        semaphore.release()
        
def run_federation(beta, overlap, gpu_id, ot, round, lr, bs, decay, split, seed, group_norm, freq, decay_fact, epochs, run_name, dataset):
    # Path to your YAML configuration and run file
    
    
    write_config_path = os.path.join(curr_dir, f'config/{run_name}.yaml')
    file_to_run = "main.py"
    read_config_path = os.path.join(curr_dir, 'config/fedml_config.yaml')
    
    torch.cuda.set_device(int(gpu_id))
     # Load the configuration
    with open(read_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Modify the configuration
    config['train_args']['batch_size'] = bs
    config['train_args']['learning_rate'] = lr
    config['train_args']['weight_decay'] = decay
    config['train_args']['comm_round'] = round
    config['train_args']['epochs'] = epochs
    config['data_args']['dataset'] = dataset
    config['common_args']['overlap_type'] = ot
    config['common_args']['alpha_dirichlet'] = beta
    config['common_args']['overlap_num'] = overlap
    config['device_args']['gpu_id'] = gpu_id
    config['common_args']['dirichlet_seed'] = seed
    config['common_args']['group_norm_size'] = group_norm
    config['common_args']['AdaptiveDecay'] = decay_fact
    config['common_args']['lr_update_freq'] = freq
    config['common_args']['data_split'] = split
    # config['common_args']['sampling_fun'] = strategy

    config['tracking_args']['run_name'] = run_name
    # Save the modified configuration
    with open(write_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    # Execute main.py
    result = subprocess.run(['python', file_to_run, '--cf', write_config_path], env=env, stderr=subprocess.PIPE)
    if result.returncode !=0:
        print("***********************************************Error message:****************************************************")
        print(f" {result.stderr} ")
        print("************************************************************************************************************")
        print(f"Error for beta: {beta}, overlap: {overlap}, seed: {seed}")
        print("************************************************************************************************************")

def check_gpu_memory(gpu_id, required_memory = 1024*1024*1024):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.free >= required_memory
                

def update_and_run_config(gpu_id, beta):
    max_processes = 50
    semaphore = Semaphore(max_processes)  # Controls the number of active processes
    processes = []
    
    # for dataset in datasets:
        # for strategy in strategies:            
            # for freq in lr_update_freqs:
                # for decay_fact in AdaptiveDecays:        
                    # for group_norm in group_norms:
                        # for bs in batch_sizes:
                            # for lr in learning_rates:
                                # for decay in decays:
                                    # for round in comm_rounds:
    
    for overlap in overlaps:
        for split in data_split:
            for ot in overlap_types:
            # for seed in seeds:                                
                # Wait if the number of active processes reaches the limit
                semaphore.acquire()
                while not check_gpu_memory(gpu_id):
                    time.sleep(120)

                p = Process(target=run_federation_with_semaphore, args=(semaphore, beta, overlap, gpu_id, ot, split, comm_rounds[0], learning_rates[0], batch_sizes[0], decays[0], seeds[0], group_norms[0], lr_update_freqs[0], AdaptiveDecays[0], datasets[0]))
                processes.append(p)
                p.start()
                time.sleep(200)

    for p in processes:
        p.join()
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run federation for a specific GPU')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU id to use (0, 1, 2, 3)')
    parser.add_argument('--beta', type=float, required=True, help='betas = [1.0, 100.0, 0.5, 0.1]')
    args = parser.parse_args()
    update_and_run_config(args.gpu_id, args.beta)
