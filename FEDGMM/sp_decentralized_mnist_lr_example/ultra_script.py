import subprocess

def run_in_tmux(gpu_id, lr):
    session_name = f"session_gpu_{gpu_id}"
    commands = [
        f"tmux new-session -d -s {session_name}",
        f"tmux send-keys -t {session_name} 'conda activate cycp' C-m",
        f"tmux send-keys -t {session_name} 'cd Federated/FEDGMM/sp_decentralized_mnist_lr_example/' C-m",
        f"tmux send-keys -t {session_name} 'python superScript.py --gpu_id {gpu_id} --lr {lr}' C-m"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    gpu_ids = [0, 1, 2, 3]
    learning_rates = [0.01, 0.005, 0.001, 0.0005]
    for gpu_id, lr in zip(gpu_ids, learning_rates):
        run_in_tmux(gpu_id, lr)
