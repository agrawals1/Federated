import yaml
import subprocess
from multiprocessing import Process
# import pynvml
import os
import argparse 
from multiprocessing import Semaphore
import torch
import time

current_file = os.path.abspath(__file__)
curr_dir = os.path.dirname(current_file)
os.chdir(curr_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
scenarios = ['linear', 'abs', 'sin', 'step']
dataHetero = [False]
datasets = ['MNIST_GMM']
# scenarios = ['sin', 'step']
# scenarios = ['linear', 'abs']
def run_federation_with_semaphore(semaphore, lr, scenario, hetero, dataset, gpu_id):
    run_name = f"Scenario:{scenario}_lr:{lr}_Heterogeneous_Clients:100_FedGMM_{dataset}" if hetero else f"Scenario:{scenario}_lr:{lr}_Homogeneous_Clients:100_FedGMM_{dataset}"
    try:
        run_federation(lr, scenario, gpu_id, hetero, run_name, dataset)
    finally:
        # Release the semaphore when the process is done
        if os.path.exists(os.path.join(curr_dir, f'config/{run_name}.yaml')):
            os.remove(os.path.join(curr_dir, f'config/{run_name}.yaml'))
        semaphore.release()
        
def run_federation(lr, scenario, gpu_id, hetero, run_name, dataset):
    # Path to your YAML configuration and run file
    write_config_path = os.path.join(curr_dir, f'config/{run_name}.yaml')
    file_to_run = "main.py"
    read_config_path = os.path.join(curr_dir, 'config/fedml_config.yaml')
    
    torch.cuda.set_device(int(gpu_id))
    # Load the configuration
    with open(read_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Modify the configuration
    config['train_args']['learning_rate'] = lr
    config['common_args']['scenario_name'] = scenario
    config['device_args']['gpu_id'] = gpu_id
    config['tracking_args']['run_name'] = run_name
    config['data_args']['data_hetero'] = hetero
    config['data_args']['dataset'] = dataset
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
        print("************************************************************************************************************")

# def check_gpu_memory(gpu_id, required_memory = 1024*1024*1024):
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     pynvml.nvmlShutdown()
#     return info.free >= required_memory
                

def update_and_run_config(gpu_id, lr):
    max_processes = 150
    semaphore = Semaphore(max_processes)  # Controls the number of active processes
    processes = []
    
    for scenario in scenarios:
        for hetero in dataHetero:
            for dataset in datasets:                                              
            # Wait if the number of active processes reaches the limit
                semaphore.acquire()
                # while not check_gpu_memory(gpu_id):
                #     time.sleep(120)
                p = Process(target=run_federation_with_semaphore, args=(semaphore, lr, scenario, hetero, dataset, gpu_id))
                processes.append(p)
                p.start()
                time.sleep(10)

    for p in processes:
        p.join()
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run federation for a specific GPU')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU id to use (0, 1, 2, 3)')
    parser.add_argument('--lr', type=float, required=True, help='learning_rates = [0.01, 0.005, 0.001, 0.0005]')
    args = parser.parse_args()
    update_and_run_config(args.gpu_id, args.lr)
