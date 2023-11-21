import yaml
import subprocess
from multiprocessing import Process
import pynvml
import time
import os
import argparse 
import torch
from multiprocessing import Semaphore


current_file = os.path.abspath(__file__)
curr_dir = os.path.dirname(current_file)
os.chdir(curr_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

batch_sizes = [32]
learning_rates = [0.1] 
decays = [1e-5]
betas = [0.3]  # parameter for dirichlet distribution
overlaps = [0] # Client overlap count
comm_rounds = [500]
# seeds = [3087732978, 918854724, 2152041540, 548193746, 993522575, 1531166731, 3136455588, 3525945833, 2018934764, 1770634816]
seeds = [3087732978]
group_norms = [0]
AdaptiveDecays = [2,4,6,8,10]
lr_update_freqs = [8]

def run_federation_with_semaphore(semaphore, beta, overlap, gpu_id, round, lr, bs, decay, seed=6967677, group_norm=0, freq=4, decay_fact=4):
    try:
        run_federation(beta, overlap, gpu_id, round, lr, bs, decay, seed, group_norm, freq, decay_fact)
    finally:
        # Release the semaphore when the process is done
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'config/fedml_config_{beta}_overlap:{overlap}_{lr}_{bs}_{decay}_{seed}.yaml')):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'config/fedml_config_{beta}_overlap:{overlap}_{lr}_{bs}_{decay}_{seed}.yaml'))
        # semaphore.release()
        
def run_federation(beta, overlap, gpu_id, round, lr, bs, decay, seed, group_norm, freq, decay_fact):
    # Path to your YAML configuration and run file
    epochs = 2 if round == 500 else (1 if round == 1000 else 4)
    run_name = f"E:{epochs}_R:{round}_Dir:{beta}_RandomSampling_AdaptiveDecay:{decay_fact}_Freq:{freq}"
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
    config['common_args']['alpha_dirichlet'] = beta
    config['common_args']['overlap_num'] = overlap
    config['device_args']['gpu_id'] = gpu_id
    config['common_args']['dirichlet_seed'] = seed
    config['common_args']['group_norm_size'] = group_norm
    config['common_args']['AdaptiveDecay'] = decay_fact
    config['common_args']['lr_update_freq'] = freq

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
                

def update_and_run_config(gpu_id):
    max_processes = 50
    semaphore = Semaphore(max_processes)  # Controls the number of active processes
    processes = []
    
    for freq in lr_update_freqs:
        for decay_fact in AdaptiveDecays:        
            for group_norm in group_norms:
                for bs in batch_sizes:
                    for lr in learning_rates:
                        for decay in decays:
                            for round in comm_rounds:
                                for beta in betas:
                                    for overlap in overlaps:
                                        for seed in seeds:                                
                                            # Wait if the number of active processes reaches the limit
                                            # semaphore.acquire()
                                            # while not check_gpu_memory(gpu_id):
                                            #     time.sleep(120)

                                            p = Process(target=run_federation_with_semaphore, args=(semaphore, beta, overlap, gpu_id, round, lr, bs, decay, seed, group_norm, freq, decay_fact))
                                            processes.append(p)
                                            p.start()
                                            # time.sleep(200)

    for p in processes:
        p.join()
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run federation for a specific GPU')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU id to use (0, 1, 2, 3)')
    args = parser.parse_args()
    update_and_run_config(args.gpu_id)
