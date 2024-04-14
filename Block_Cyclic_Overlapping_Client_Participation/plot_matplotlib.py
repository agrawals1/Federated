import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D 
from scipy.interpolate import make_interp_spline
import numpy as np

dir = 100.0
def plot_global_test_accuracy_vs_cycle(file_path, runs_Clients, runs_Groups):
    dataframe = pd.read_csv(file_path)
    
    
    # Define colors and linestyles
    colors = ['#FF0000', '#0000FF', '#FF7F00', '#FF00FF']  # More colors for different overlaps  # Define more colors if you have more than 7 different overlaps
    linestyles = ['--', '-']  # Solid line for groups, dashed line for clients

    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 9), dpi=300, gridspec_kw={'height_ratios': [10000, 1], 'hspace': 0.2})
    
    overlap_annotations = [] 

    # Function to plot each run with specific style
    def plot_run(ax, run_data, color, linestyle, width=2.0, marker=None):
    # Generate 300 points for the x-axis (smoothness factor) between min and max cycle values
        x_smooth = np.linspace(run_data['cycle'].min(), run_data['cycle'].max(), 30)
    
        # Interpolation function creation
        spline = make_interp_spline(run_data['cycle'], run_data['global_test_Acc'], k=3)  # k=3 for cubic spline
        y_smooth = spline(x_smooth)

        ax.plot(x_smooth, y_smooth, linewidth=width, color=color, linestyle=linestyle, marker=marker)

    for i, run_name in enumerate(runs_Groups):
        run_data = dataframe[dataframe['run_name'] == run_name]
        run_data = run_data.dropna(subset=['global_test_Acc'])
        run_data['cycle'] = run_data['cycle'].astype(float)
        run_data['global_test_Acc'] = run_data['global_test_Acc'].astype(float)
        overlap = run_name[run_name.find('Overlap:') + len('Overlap:')]
        color = colors[i % len(colors)] if overlap!='0' else '#000000'
        linestyle = linestyles[1] if overlap!='0' else ':'
        width = 6.0 if overlap == '0' else 2.0
        plot_run(ax2, run_data, color, linestyle, width)
        last_point = run_data.iloc[-1]  # Get the last data point
        overlap_annotations.append((last_point['cycle'], last_point['global_test_Loss'], overlap))
        

    for i, run_name in enumerate(runs_Clients):
        run_data = dataframe[dataframe['run_name'] == run_name]
        run_data = run_data.dropna(subset=['global_test_Acc'])
        run_data['cycle'] = run_data['cycle'].astype(float)
        run_data['global_test_Acc'] = run_data['global_test_Acc'].astype(float)
        overlap = run_name[run_name.find('Overlap:') + len('Overlap:')]
        color = colors[(i+1) % len(colors)]
        plot_run(ax2, run_data, color, linestyles[0], marker=None)
        last_point = run_data.iloc[-1]  # Get the last data point
        overlap_annotations.append((last_point['cycle'], last_point['global_test_Acc'], overlap))
        
        
    overlap_annotations.sort(key=lambda x: x[1])  # Sort by y-value
    annotated_y_values = []
    for x, y, overlap in overlap_annotations:
    # If the list is empty or the last y-value is sufficiently far from the current y-value, simply add the new overlap.
        if not annotated_y_values or y - annotated_y_values[-1][1] >= 0.0018:
            annotated_y_values.append((x, y, overlap))
        else:
            # Merge current overlap with the last one's overlaps due to closeness.
            last_x, last_y, last_overlaps = annotated_y_values.pop()
            combined_overlaps = f"{overlap}, {last_overlaps}"
            annotated_y_values.append((last_x, max(y, last_y), combined_overlaps))

    for x, y, overlap in annotated_y_values:
        ax2.text(x + 2, y, f"({overlap})", fontsize=25, color='black')  # Adjust color as needed
    ax1.set_ylim(0)
    # ax2.set_ylim(0.41, 0.46) # 0.1
    # ax2.set_ylim(0.48, 0.503) # 0.5
    # ax2.set_ylim(0.48, 0.509) # 1.0, 100.0
    ax2.set_ylim(19000, 26000) # 100.0 clientsVSgroups
    # ax2.set_ylim(0.47, 0.521) # 1.0 clientsVSgroups
    # ax2.set_ylim(0.47, 0.51) # 0.5 clientsVSgroups
    # ax2.set_ylim(0.41, 0.452) # 0.1 clientsVSgroups

    ax1.set_yticks([0.0])
    ax1.set_yticklabels([0.0], fontsize=30)
    # ax2.set_yticks([0.41, 0.42, 0.43, 0.44, 0.45]) # 0.1
    # ax2.set_yticks([0.48, 0.49, 0.50]) # 0.5 1.0, 100.0
    ax2.set_yticks([19000, 22000, 25000]) # 100.0 clientsVSgroups
    # ax2.set_yticks([0.47, 0.49, 0.51, 0.52]) # 1.0 clientsVSgroups
    # ax2.set_yticks([0.47, 0.48, 0.49, 0.50]) # 0.5 clientsVSgroups
    # ax2.set_yticklabels([0.41, 0.42, 0.43, 0.44, 0.45], fontsize=30) # 0.1
    ax2.set_yticklabels([19000, 22000, 25000], fontsize=30) # 100.0
    # ax2.set_yticklabels([0.47, 0.49, 0.51, 0.52], fontsize=30) # 1.0 clientsVSgroups
    # ax2.set_yticklabels([0.47, 0.48, 0.49, 0.50], fontsize=30) # 0.5 clientsVSgroups

    plt.xlabel('Cycle', fontsize=30)
    ax2.set_ylabel('Global Test Accuracy', fontsize=30)
    ax2.grid(True)
    ax1.grid(True)

    ax1.set_xticks([0, 40, 80, 120, 160])
    ax1.set_xticklabels([0, 40, 80, 120, 160], fontsize=30)
    custom_lines = [Line2D([0], [0], color='black', linewidth=3, linestyle='--', marker=None),
                Line2D([0], [0], color='black', linewidth=3, linestyle='-', marker=None)]
    fig.legend(custom_lines, ['DataSPlit', 'FullData'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=30)

    plt.savefig(f"Federated/Block_Cyclic_Overlapping_Client_Participation/Dir{dir}SplitVsNoSplit_CIFAR10.png")
    
def plot_global_test_Loss_vs_cycle(file_path, runs_Clients, runs_Groups):
    dataframe = pd.read_csv(file_path)
    
    
    # Define colors and linestyles
    colors = ['#FF0000', '#0000FF', '#FF7F00', '#FF00FF']  # More colors for different overlaps  # Define more colors if you have more than 7 different overlaps
    linestyles = ['--', '-']  # Solid line for groups, dashed line for clients

    fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 9), dpi=300, gridspec_kw={'height_ratios': [10000, 1], 'hspace': 0.2})
    
    overlap_annotations = [] 

    # Function to plot each run with specific style
    def plot_run(ax, run_data, color, linestyle, width=2.0, marker=None):
    # Generate 300 points for the x-axis (smoothness factor) between min and max cycle values
        x_smooth = np.linspace(run_data['cycle'].min(), run_data['cycle'].max(), 30)
    
        # Interpolation function creation
        spline = make_interp_spline(run_data['cycle'], run_data['global_test_Loss'], k=3)  # k=3 for cubic spline
        y_smooth = spline(x_smooth)

        ax.plot(x_smooth, y_smooth, linewidth=width, color=color, linestyle=linestyle, marker=marker)

    for i, run_name in enumerate(runs_Groups):
        run_data = dataframe[dataframe['run_name'] == run_name]
        run_data = run_data.dropna(subset=['global_test_Loss'])
        run_data['cycle'] = run_data['cycle'].astype(float)
        run_data['global_test_Loss'] = run_data['global_test_Loss'].astype(float)
        overlap = run_name[run_name.find('Overlap:') + len('Overlap:')]
        color = colors[i % len(colors)] if overlap!='0' else '#000000'
        linestyle = linestyles[1] if overlap!='0' else ':'
        width = 6.0 if overlap == '0' else 2.0
        plot_run(ax2, run_data, color, linestyle, width)
        last_point = run_data.iloc[-1]  # Get the last data point
        overlap_annotations.append((last_point['cycle'], last_point['global_test_Loss'], overlap))
        

    for i, run_name in enumerate(runs_Clients):
        run_data = dataframe[dataframe['run_name'] == run_name]
        run_data = run_data.dropna(subset=['global_test_Loss'])
        run_data['cycle'] = run_data['cycle'].astype(float)
        run_data['global_test_Loss'] = run_data['global_test_Loss'].astype(float)
        overlap = run_name[run_name.find('Overlap:') + len('Overlap:')]
        color = colors[(i+1) % len(colors)]
        plot_run(ax2, run_data, color, linestyles[0], marker=None)
        last_point = run_data.iloc[-1]  # Get the last data point
        overlap_annotations.append((last_point['cycle'], last_point['global_test_Loss'], overlap))
        
        
    overlap_annotations.sort(key=lambda x: x[1])  # Sort by y-value
    annotated_y_values = []
    for x, y, overlap in overlap_annotations:
    # If the list is empty or the last y-value is sufficiently far from the current y-value, simply add the new overlap.
        if not annotated_y_values or y - annotated_y_values[-1][1] >= 500:
            annotated_y_values.append((x, y, overlap))
        else:
            # Merge current overlap with the last one's overlaps due to closeness.
            last_x, last_y, last_overlaps = annotated_y_values.pop()
            combined_overlaps = f"{last_overlaps}, {overlap}"
            annotated_y_values.append((last_x, max(y, last_y), combined_overlaps))

    for x, y, overlap in annotated_y_values:
        ax2.text(x + 2, y, f"({overlap})", fontsize=25, color='black')  # Adjust color as needed
    ax1.set_ylim(0)
    # ax2.set_ylim(0.41, 0.46) # 0.1
    # ax2.set_ylim(0.48, 0.503) # 0.5
    # ax2.set_ylim(0.48, 0.509) # 1.0, 100.0
    ax2.set_ylim(13000, 25000) # 100.0 clientsVSgroups
    # ax2.set_ylim(0.47, 0.521) # 1.0 clientsVSgroups
    # ax2.set_ylim(0.47, 0.51) # 0.5 clientsVSgroups
    # ax2.set_ylim(0.41, 0.452) # 0.1 clientsVSgroups

    ax1.set_yticks([0.0])
    ax1.set_yticklabels([0.0], fontsize=30)
    # ax2.set_yticks([0.41, 0.42, 0.43, 0.44, 0.45]) # 0.1
    # ax2.set_yticks([0.48, 0.49, 0.50]) # 0.5 1.0, 100.0
    ax2.set_yticks([13000, 15000, 17000, 19000]) # 100.0 clientsVSgroups
    # ax2.set_yticks([0.47, 0.49, 0.51, 0.52]) # 1.0 clientsVSgroups
    # ax2.set_yticks([0.47, 0.48, 0.49, 0.50]) # 0.5 clientsVSgroups
    # ax2.set_yticklabels([0.41, 0.42, 0.43, 0.44, 0.45], fontsize=30) # 0.1
    ax2.set_yticklabels([13000, 15000, 17000, 19000], fontsize=30) # 100.0
    # ax2.set_yticklabels([0.47, 0.49, 0.51, 0.52], fontsize=30) # 1.0 clientsVSgroups
    # ax2.set_yticklabels([0.47, 0.48, 0.49, 0.50], fontsize=30) # 0.5 clientsVSgroups

    plt.xlabel('Cycle', fontsize=30)
    ax2.set_ylabel('Global Test Loss', fontsize=30)
    ax2.grid(True)
    ax1.grid(True)

    ax1.set_xticks([0, 40, 80, 120, 160])
    ax1.set_xticklabels([0, 40, 80, 120, 160], fontsize=30)
    custom_lines = [Line2D([0], [0], color='black', linewidth=3, linestyle='--', marker=None),
                Line2D([0], [0], color='black', linewidth=3, linestyle='-', marker=None)]
    fig.legend(custom_lines, ['DataSplit', 'FullData'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=30)

    plt.savefig(f"Federated/Block_Cyclic_Overlapping_Client_Participation/Dir{dir}SplitVsNoSplitLoss_CIFAR10.png")

# Example function call

runs_FullData = [f"Dir:{dir}_Overlap:{olap}_Remark:Complete_Data" for olap in [0,3,6,9]]
runs_DataSplit = [f"Dir:{dir}_Overlap:{olap}_Remark:Data_Split" for olap in [3,6,9]]
plot_global_test_Loss_vs_cycle('SplitVsNoSplit_CIFAR10.csv', runs_DataSplit, runs_FullData)







