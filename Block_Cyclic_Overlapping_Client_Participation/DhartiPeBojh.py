# # # import os
# # # import subprocess
# # # import sys
import random
# # # def get_process_info(pid):
# # #     """
# # #     Get the command line and start time of a process given its PID.
# # #     """
# # #     try:
# # #         # Using 'ps' to get the command and start time of the process
# # #         output = subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'cmd='])
# # #         start_time, command = output.decode().strip().split(maxsplit=1)
# # #         return command, start_time
# # #     except subprocess.CalledProcessError:
# # #         print(f"Process with PID {pid} not found.")
# # #         return None, None

# # # def kill_process(pid):
# # #     """
# # #     Kill a process given its PID.
# # #     """
# # #     try:
# # #         os.kill(pid, 9)  # Using signal 9 for forceful termination
# # #         print(f"Process with PID {pid} has been killed.")
# # #     except Exception as e:
# # #         print(f"Failed to kill process {pid}: {e}")

# # # def process_handler(pids):
# # #     """
# # #     Main function to handle the processes: fetching info and killing duplicates.
# # #     """
# # #     process_info = {}
# # #     for pid in pids:
# # #         command, start_time = get_process_info(pid)
# # #         start_time = command[7:15]
# # #         command = command[21:]
# # #         if command:
# # #             # Checking if the command was already seen
# # #             if command in process_info:
# # #                 existing_pid, existing_start_time = process_info[command]
# # #                 # Kill the process that was started later
# # #                 if start_time > existing_start_time:
# # #                     kill_process(pid)
# # #                 else:
# # #                     kill_process(existing_pid)
# # #                     process_info[command] = (pid, start_time)
# # #             else:
# # #                 process_info[command] = (pid, start_time)

# # # if __name__ == "__main__":
# # #     process_ids = [51651, 51655, 52754, 52758, 54239, 54243, 54765, 54769, 55285, 55289, 51925, 51929, 52866, 52870, 54363, 54367, 54881, 54885, 55397, 55401, 52239, 52243, 53604, 53608, 54475, 54479, 55028, 55032, 55575, 55579]  # Replace with actual PIDs
# # #     process_handler(process_ids)

# # # # import psutil

# # # # def get_process_commands(process_ids):
# # # #     processes = []
# # # #     for pid in process_ids:
# # # #         try:
# # # #             p = psutil.Process(pid)
# # # #             processes.append((pid, p.create_time(), p.cmdline()))
# # # #         except psutil.NoSuchProcess:
# # # #             print(f"No process found with PID {pid}")

# # # #     # Sort the processes by creation time (the second element of the tuple)
# # # #     processes.sort(key=lambda x: x[1])

# # # #     # Format the output
# # # #     result = [{"Pid": pid, "Command": " ".join(cmd)} for pid, _, cmd in processes]
# # # #     return result

# # # # # Example usage
# # # # process_ids = [51651, 51655, 52754, 52758, 54239, 54243, 54765, 54769, 55285, 55289, 51925, 51929, 52866, 52870, 54363, 54367, 54881, 54885, 55397, 55401, 52239, 52243, 53604, 53608, 54475, 54479, 55028, 55032, 55575, 55579]  # Replace with actual PIDs
# # # # print(get_process_commands(process_ids))

# # import psutil

# # # Sample data as input
# # process_data = """
# # 87162 C python 8498MiB
# # 88422 C python 7586MiB
# # 89223 C python 6666MiB
# # 89924 C python 5754MiB
# # 90488 C python 5302MiB
# # 51651 C python 260MiB
# # 51655 C python 9408MiB
# # 54243 C python 8498MiB
# # 55289 C python 7586MiB
# # 86324 C python 260MiB
# # 86328 C python 9398MiB
# # 87598 C python 260MiB
# # 87612 C python 8484MiB
# # 51925 C python 260MiB
# # 55401 C python 7572MiB
# # 86827 C python 260MiB
# # 86834 C python 9424MiB
# # 88248 C python 260MiB
# # 88252 C python 8504MiB
# # 89073 C python 260MiB
# # 89085 C python 7594MiB
# # 89762 C python 260MiB
# # 89766 C python 6674MiB
# # 90354 C python 260MiB
# # 90358 C python 5762MiB
# # 52239 C python 260MiB
# # 52243 C python 9424MiB
# # 54479 C python 8504MiB
# # 55579 C python 7594MiB
# # """

# # # Parsing the data
# # process_lines = process_data.strip().split("\n")
# # process_info = []

# # # Collecting data
# # for line in process_lines:
# #     pid, _, _, memory = line.split()  # Adjust based on the actual format
# #     process_info.append((pid, memory))

# # # Fetching owner names
# # results = []
# # for pid, memory in process_info:
# #     try:
# #         process = psutil.Process(int(pid))
# #         user = process.username()
# #         results.append(f"{pid} | {memory} | {user}")
# #     except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
# #         results.append(f"{pid} | {memory} | Process not found or access denied")

# # # Output
# # for result in results:
# #     print(result)
# import psutil
# import GPUtil

# # Mock data, replace this with your actual process IDs and memory usage
# process_ids = [
#     87162, 88422, 89223, 89924, 90488,
#     51651, 51655, 54243, 55289, 86324,
#     86328, 87598, 87612, 51925, 55401,
#     86827, 86834, 88248, 88252, 89073,
#     89085, 89762, 89766, 90354, 90358,
#     52239, 52243, 54479, 55579
# ]

# # Function to get CPU and GPU utilization
# def get_process_utilization(pids):
#     gpus = GPUtil.getGPUs()
#     utilization_info = []

#     for pid in pids:
#         try:
#             process = psutil.Process(pid)
#             cpu_usage = process.cpu_percent(interval=1) / psutil.cpu_count()  # Normalize by CPU count
#             # Sum GPU usage from all GPUs where the process is found
#             # gpu_usage = sum([gpu.load for gpu in gpus if gpu.id in process.gpu_ids()])
#             gpu_usage = None

#             utilization_info.append({
#                 'pid': pid,
#                 'cpu_usage': cpu_usage,
#                 'gpu_usage': gpu_usage
#             })

#         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
#             utilization_info.append({
#                 'pid': pid,
#                 'cpu_usage': 'N/A',
#                 'gpu_usage': 'N/A'
#             })

#     return utilization_info

# # Get the utilization
# utilization_data = get_process_utilization(process_ids)

# # Display the results
# print("PID\tCPU Usage (%)\tGPU Usage (%)")
# for data in utilization_data:
#     print(f"{data['pid']}\t{data['cpu_usage']}\t{data['gpu_usage']}")
# # After collecting the utilization data, insert the following code

# for data in utilization_data:
#     if data['cpu_usage'] == 0.0:
#         try:
#             process = psutil.Process(data['pid'])
#             process.terminate()  # or process.kill() if you want to force kill
#             print(f"Terminated process with PID {data['pid']} due to 0.0% CPU utilization.")
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             print(f"Could not terminate process {data['pid']}: No such process or access denied.")


# # Identify "bad" processes (example criteria: GPU memory is high but CPU and GPU usage are low)
# for data in utilization_data:
#     if isinstance(data['cpu_usage'], float) and isinstance(data['gpu_usage'], float):
#         if data['cpu_usage'] < 10 and data['gpu_usage'] < 0.1:  # Adjust thresholds as necessary
#             print(f"Bad process: {data['pid']} - Low CPU and GPU usage")
# Re-run the function with correct logic based on the described scenario
def generate_cycle_more_groups_same_size(client_num_in_total, total_groups, common_clients_num):
    base_group_size = client_num_in_total // total_groups
    clients = list(range(client_num_in_total))
    reserve = []

    # Create initial K groups with M/K clients each
    groups = [clients[i * base_group_size:(i + 1) * base_group_size] for i in range(total_groups)]

    # Select t common clients from the first group
    common_clients = random.sample(groups[0], common_clients_num)

    # Shift t clients from each group (except the first one) to the reserve and add t common clients
    for i in range(1, total_groups):
        shifted_clients = random.sample(groups[i], common_clients_num)
        groups[i] = [client for client in groups[i] if client not in shifted_clients]
        groups[i].extend(common_clients)
        reserve.extend(shifted_clients)

    # Create new groups with the reserve clients
    while reserve:
        new_group = reserve[:base_group_size - common_clients_num]
        reserve = reserve[base_group_size - common_clients_num:]
        new_group.extend(common_clients)
        groups.append(new_group)

    return groups

# Test values
M = 12  # Total clients
K = 4   # Initial total groups
t = 2   # Common clients number

# Generate the cycle
cycle_more_groups_same_size = generate_cycle_more_groups_same_size(M, K, t)
print(cycle_more_groups_same_size)
    