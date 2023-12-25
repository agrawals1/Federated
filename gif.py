# Adjusting parameters for the final tweaks
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
clients_per_round = 10  # Reducing the number of clients participating in a round to 4
radius = 0.8  # Reduced radius
total_rounds = 20
total_clients = 30
overlap_num = 3
frames_dir = "/home/shubham/Federated/Client_Participation GIF"
fps = 2/3
# Recreate the frames with the new specification

def client_sampling_cyclic_overlap_pattern_with_color(round_idx, client_num_in_total, client_num_per_round, overlap_num):
    current_clients = set((round_idx * (client_num_per_round - overlap_num) + i) % client_num_in_total for i in range(client_num_per_round))
    previous_clients = set(((round_idx - 1) * (client_num_per_round - overlap_num) + i) % client_num_in_total for i in range(client_num_per_round))
    overlapping_clients = current_clients.intersection(previous_clients)
    non_overlapping_clients = current_clients - overlapping_clients
    return overlapping_clients, non_overlapping_clients


for round_idx in range(total_rounds):
    overlapping_clients, non_overlapping_clients = client_sampling_cyclic_overlap_pattern_with_color(
        round_idx, total_clients, clients_per_round, overlap_num
    )
    fig, ax = plt.subplots()

    # Represent each client as a point on a reduced radius circle
    theta = np.linspace(0, 2*np.pi, total_clients, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Plot overlapping clients in green, non-overlapping clients in red, and others in gray
    for client in range(total_clients):
        color = 'gray'  # Default color for inactive clients
        if client in overlapping_clients:
            color = 'green'
        elif client in non_overlapping_clients:
            color = 'red'
        ax.text(x[client], y[client], str(client), ha='center', va='center', bbox=dict(facecolor=color, alpha=0.5))

    # Set labels and title
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(f'Round {round_idx+1}')

    # Save the frame
    plt.savefig(f"{frames_dir}/frame_{round_idx:03d}.png")
    plt.close()

# Recreate GIF with updated frames, ensuring it loops indefinitely
frames = [imageio.imread(f"{frames_dir}/frame_{i:03d}.png") for i in range(total_rounds)]
gif_path_circular_final = f"/home/shubham/Federated/Client_Participation GIF/federated_learning_animation_circular_final_{overlap_num}.gif"
imageio.mimsave(gif_path_circular_final, frames, fps=fps, loop=0)  # loop=0 for infinite looping

gif_path_circular_final  # Path to the final circular GIF

