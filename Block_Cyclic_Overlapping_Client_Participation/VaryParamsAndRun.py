import yaml
import subprocess
from multiprocessing import Process
import pynvml
import time
import os
import argparse 
import torch
import random

current_file = os.path.abspath(__file__)
curr_dir = os.path.dirname(current_file)
os.chdir(curr_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

betas = [0.05, 0.2, 0.7]
overlaps = [0, 1, 3, 5, 7, 9]
seeds = [random.randint(0, 2**32-1) for _ in range(10)]
comm_rounds = [250, 1000]
params = {"betas": betas, "overlaps": overlaps, "seeds" : seeds, "comm_rounds": comm_rounds}

def run_federation(beta, overlap, seed, gpu_id, round):
    # Path to your YAML configuration and run file
    write_config_path = f'fedml_config_{beta}_{overlap}_{seed}.yaml'
    file_to_run = "main.py"
    read_config_path = 'fedml_config.yaml'
  
    torch.cuda.set_device(int(gpu_id))
     # Load the configuration
    with open(read_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Modify the configuration
    config['train_args']['comm_round'] = round
    config['train_args']['epochs'] = (1 if round == 1000 else 4)
    config['common_args']['alpha_dirichlet'] = beta
    config['common_args']['overlap_num'] = overlap
    config['device_args']['gpu_id'] = gpu_id
    config['common_args']['dirichlet_seed'] = seed
    config['tracking_args']['run_name'] = f"E1_R1000_LRe-1_Dir:{beta}_Overlap:{overlap}_Seed:{seed}_batch64"
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
                

def update_and_run_config(params, gpu_id):
    processes = []
    for seed in params["seeds"]:
        for round in params["comm_rounds"]:
            for beta in params["betas"]:
                for overlap in params["overlaps"]:                    
                    while not check_gpu_memory(gpu_id):
                        time.sleep(120)
                    p = Process(target=run_federation, args=(beta, overlap, seed, gpu_id, round))
                    processes.append(p)
                    p.start()
                    time.sleep(80)

    for p in processes:
        p.join()
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run federation for a specific GPU')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU id to use (0 or 1)')
    args = parser.parse_args()
    update_and_run_config(params, args.gpu_id)
