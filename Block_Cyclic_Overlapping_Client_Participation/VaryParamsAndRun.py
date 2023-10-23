import yaml
import subprocess

def update_and_run_config(betas, overlaps):
    # Path to your YAML configuration file
    config_path = 'fedml_config.yaml'
    file_to_run = "main.py"
    for beta in betas:
        for overlap in overlaps:
            # Load the configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Modify the configuration
            config['common_args']['alpha_dirichlet'] = beta
            config['common_args']['overlap_num'] = overlap
            config['tracking_args']['run_name'] = f"E4_R75_LRe-1_Dir:{beta}_Overlap:{overlap}"
            # Save the modified configuration
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            # Execute main.py
            subprocess.run(['python', file_to_run, '--cf', config_path])


# Define the variations for learning_rate and batch_size
betas = [1, 0.1, 0.05, 0.01]
overlaps = [0, 1, 3, 5, 7, 9]

update_and_run_config(betas, overlaps)
