import pandas as pd
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
project_path = "shubham22124/SplitVsNoSplitFL"

# Define the filter condition, for example, fetching runs with specific names
# You can change "run-name-1", "run-name-2" to the actual run names you want to fetch
# filter_condition = {"$or": [{"name": "run-name-1"}, {"name": "run-name-2"}]}
filter_condition = filter_condition = {"config.dataset": "CIFAR10"}
runs = api.runs(project_path, filters=filter_condition)

all_runs_data = []

for run in runs:
    # Fetch historical data for each run using scan_history
    history_items = run.scan_history()  # Fetches all metrics logged at each step

    # Convert history items to DataFrame
    history_df = pd.DataFrame([item for item in history_items])

    # Include run's name or ID in the dataframe for identification
    history_df['run_id'] = run.id
    history_df['run_name'] = run.name

    all_runs_data.append(history_df)

# Concatenate all individual run dataframes into a single dataframe
full_df = pd.concat(all_runs_data, ignore_index=True)

# Save the combined dataframe to a CSV file
full_df.to_csv("SplitVsNoSplit_CIFAR10.csv")
