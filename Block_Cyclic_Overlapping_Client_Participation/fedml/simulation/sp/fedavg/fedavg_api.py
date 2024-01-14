import copy
import logging
import torch
import math
import wandb
from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from collections import OrderedDict
from ClientSampler import ClientSampler
import copy
from ..fedopt.optrepo import OptRepo
import os
current_dir = os.path.dirname(__file__)
base_dir = os.path.join(current_dir, "../../../..")  # Adjust based on the relative path
models_dir = os.path.join(base_dir, "models")

class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_sampler = ClientSampler(args)
        # self.sampling_functions = {"_client_sampling": self.client_sampler._client_sampling, 
        #                       "client_sampling_cyclic_overlap_pattern": self.client_sampler.client_sampling_cyclic_overlap_pattern,
        #                       "client_sampling_cyclic_noOverlap_random": self.client_sampler.client_sampling_cyclic_noOverlap_random,
        #                       "client_sampling_cyclic_overlap_random": self.client_sampler.client_sampling_cyclic_overlap_random,
        #                       "all_participate": self.client_sampler.all_participate
        #                       }
        
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model,copy.deepcopy(model),args)
        self._instanciate_opt()
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        self.participation_counts = {client_id: 0 for client_id in range(self.args.client_num_in_total)}
        self.group_ids = {client_id: [] for client_id in range(self.args.client_num_in_total)}
        self.cycle = self.generate_cycle(self.args.client_num_in_total, self.args.total_groups, self.args.overlap_num)
        self.count_participations(self.cycle, self.args.client_num_in_total)
        self.currently_part_in = {client_id: 0 for client_id in range(self.args.client_num_in_total)}
        self.total_cycles = math.ceil(self.args.comm_round / len(self.cycle)) if not self.args.total_cycles else self.args.total_cycles
        if self.args.group_wise_models:
            self.group_specific_models = {group_id: copy.deepcopy(model) for group_id in range(len(self.cycle))}
        
        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, copy.deepcopy(self.model_trainer),
        )
    
    def update_group_wise_model(self, w_global, group_id, cycle_idx):            
        w_group = self.group_specific_models[group_id].state_dict()
        updated_w_group = {}
        if not cycle_idx:
            for key in w_global.keys():
                updated_w_group[key] = w_global[key] 
        else:
            
            for key in w_global.keys():
                updated_w_group[key] = (w_global[key] / (cycle_idx + 1)) + w_group[key] * (cycle_idx / (cycle_idx + 1))
        model_path = os.path.join(models_dir, self.args.run_name, f'updated_w_group_{group_id}.pt')
        torch.save(updated_w_group, model_path)
        self.group_specific_models[group_id].load_state_dict(updated_w_group)
        
    def _set_model_global_grads(self, new_state):
        new_model = copy.deepcopy(self.model_trainer.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(self.model_trainer.model.parameters(), new_model.parameters()):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.model_trainer.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.model_trainer.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        self.model_trainer.set_model_params(new_model_state_dict)
    
    
    def _instanciate_opt(self):
        self.opt = OptRepo.name2cls(self.args.server_optimizer)(
            # self.model_global.parameters(), lr=self.args.server_lr
            self.model_trainer.model.parameters(),
            lr=self.args.server_lr,
            # momentum=0.9 # for fedavgm
            eps = 1e-3 #for adaptive optimizer
        )
    
    def update_cycle_wise_metrics(self, round_wise_metrics_local, round_wise_metrics_group, round_wise_metrics_global, cycle_wise_metrics):
        train_metrics_local, test_metrics_local = round_wise_metrics_local
        train_metrics_group, test_metrics_group = round_wise_metrics_group
        test_metrics_global = round_wise_metrics_global
        # Update local metrics
        for key in ['num_samples', 'num_correct', 'losses']:
            cycle_wise_metrics["cycle_wise_train_metrics_local"][key].extend(train_metrics_local[key])
            cycle_wise_metrics["cycle_wise_test_metrics_local"][key].extend(test_metrics_local[key])

        # Update group metrics
        for key in ['num_samples', 'num_correct', 'losses']:
            cycle_wise_metrics["cycle_wise_train_metrics_group"][key].extend(train_metrics_group[key])
            cycle_wise_metrics["cycle_wise_test_metrics_group"][key].extend(test_metrics_group[key])
            
        for key1, key2 in {'num_samples':'test_total', 'num_correct':'test_correct', 'losses':'test_loss'}.items():
            cycle_wise_metrics["cycle_wise_test_metrics_global"][key1].append(test_metrics_global[key2])

        return cycle_wise_metrics    
                    
    def count_participations(self, cycle, client_num_in_total):
       
        # Count participations for each client in the cycle
        for group_id, client_group in enumerate(cycle):
            for client_id in client_group:
                self.participation_counts[client_id] += 1
                self.group_ids[client_id].append(group_id)

    
    def generate_cycle(self, client_num_in_total, total_groups, overlap_num):
        cycle = []
        start_idx = 0
        
        base_grp_size = client_num_in_total // total_groups

        for _ in range(total_groups):
            self.group_size = base_grp_size + overlap_num

            end_idx = start_idx + self.group_size

            if end_idx > client_num_in_total:
                end_idx = client_num_in_total

            client_indices = list(range(start_idx, end_idx))
            cycle.append(client_indices)           
            
            if end_idx >= client_num_in_total:
                break
            start_idx += base_grp_size

        return cycle 

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    
    
    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        if self.args.active:

            for client_idx in range(self.args.client_num_in_total):
                c = Client(
                    client_idx,
                    train_data_local_dict[client_idx],
                    test_data_local_dict[client_idx],
                    train_data_local_num_dict[client_idx],
                    self.args,
                    self.device,
                    model_trainer,
                    group_id = self.group_ids[client_idx]
                )
                self.client_list.append(c)
        else:
            for client_idx in range(self.group_size):
                c = Client(
                    client_idx,
                    train_data_local_dict[client_idx],
                    test_data_local_dict[client_idx],
                    train_data_local_num_dict[client_idx],
                    self.args,
                    self.device,
                    model_trainer,
                    group_id = self.group_ids[client_idx]
                )
                self.client_list.append(c)
        logging.info("############setup_clients (END)#############")


    def sub_ordered_dict(self,dict1, dict2):
        result = OrderedDict()
        for key, val in dict1.items():
            result[key] = (val - dict2[key])
        return result
    
    def add_ordered_dict(self, alpha, dict1, dict2):
        result = OrderedDict()
        for key, val in dict1.items():
            result[key] = (val + alpha*dict2[key])
        return result
    

    def train(self):
        logging.info("self.model_trainer = %s", self.model_trainer)
        w_global = self.model_trainer.get_model_params()
        
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        marker = 0
        initial_rounds = self.args.comm_round
        decay_factor = self.args.AdaptiveDecay
        update_frequency = self.args.lr_update_freq
        threshold = (initial_rounds - marker) / update_frequency
        
        for round_idx in range(self.args.comm_round):
            wandb.log({"learning_rate": self.args.learning_rate, "round": round_idx}, step=round_idx)
            if (round_idx+1) >= threshold:
                marker = threshold
                threshold = marker + (initial_rounds-marker) / update_frequency
                self.args.learning_rate = (self.args.learning_rate / decay_factor) if self.args.learning_rate > 0.00001 else self.args.learning_rate
                
            logging.info("################Communication round : %s", round_idx)

            w_locals = self._train_clients_for_round(round_idx, w_global)

            # Update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            # Test results based on conditions
            self._test_models_based_on_conditions(round_idx, w_global)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()
        
    def train_cycle(self):
        logging.info("self.model_trainer = %s", self.model_trainer)
        w_global = self.model_trainer.get_model_params()
        
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        
        round_idx = 0
        marker = 0
        initial_cycles = self.args.total_cycles
        decay_factor = self.args.AdaptiveDecay
        update_frequency = self.args.lr_update_freq
        threshold = (initial_cycles - marker) / update_frequency
        for cycle_idx in range(self.total_cycles):          
            cycle_wise_metrics = {"cycle_wise_train_metrics_local": {"num_samples": [], "num_correct": [], "losses": []}, "cycle_wise_test_metrics_local": {"num_samples": [], "num_correct": [], "losses": []}, 
                                  "cycle_wise_train_metrics_group": {"num_samples": [], "num_correct": [], "losses": []}, "cycle_wise_test_metrics_group": {"num_samples": [], "num_correct": [], "losses": []},
                                  "cycle_wise_test_metrics_global": {"num_samples": [], "num_correct": [], "losses": []}}
            logging.info("Starting Cycle %d", cycle_idx)
            wandb.log({"cycle": cycle_idx, "round": round_idx})
            for group_id, client_group in enumerate(self.cycle):
                if (cycle_idx+1) >= threshold:
                    marker = threshold
                    threshold = marker + (initial_cycles-marker) / update_frequency
                    self.args.learning_rate = (self.args.learning_rate / decay_factor) if self.args.learning_rate > 0.00001 else self.args.learning_rate
                wandb.log({"learning_rate": self.args.learning_rate, "round": round_idx})
                logging.info("Communication Round %d in Cycle %d", round_idx, cycle_idx)
                mlops.log_round_info(self.args.comm_round, round_idx)

                # Train clients in the current group
                w_locals = self._train_clients_for_group(client_group, w_global)

                # Update global weights
                # reset weight after standalone simulation
                self.model_trainer.set_model_params(w_global)
                # update global weights
                w_avg = self._aggregate(w_locals)
                # server optimizer
                self.opt.zero_grad()
                opt_state = self.opt.state_dict()
                self._set_model_global_grads(w_avg)
                self._instanciate_opt()
                self.opt.load_state_dict(opt_state)
                self.opt.step()
                w_global = self.model_trainer.get_model_params()
                model_path = os.path.join(models_dir, self.args.run_name, f'updated_w_global.pt')
                os.makedirs(os.path.join(models_dir, self.args.run_name), exist_ok=True)
                torch.save(w_global, model_path)
                # Test models
                round_wise_metrics_local = self._local_test_on_participating_clients(round_idx, True)
                if self.args.group_wise_models:
                    self.update_group_wise_model(w_global, group_id, cycle_idx)    
                    round_wise_metrics_group = self._test_group_wise_model_on_local_data(round_idx, True)                
                round_wise_metrics_global = self._test_global_model_on_global_data(w_global, round_idx, True)
                cycle_wise_metrics = self.update_cycle_wise_metrics(round_wise_metrics_local, round_wise_metrics_group, round_wise_metrics_global, cycle_wise_metrics)
                round_idx += 1
            self.currently_part_in = {client_id: 0 for client_id in range(self.args.client_num_in_total)}
            self._test_cycle_wise(cycle_wise_metrics, cycle_idx)
        mlops.log_round_info(self.args.comm_round, -1)
        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    def _train_clients_for_group(self, client_group, w_global):
        w_locals = []
        for idx, client_id in enumerate(client_group):
            client = self.client_list[idx]
            client.update_client(
                    client_id,
                    self.train_data_local_dict[client_id],
                    self.test_data_local_dict[client_id],
                    self.train_data_local_num_dict[client_id],
                    self.group_ids[client_id]
                )
            part_cnt = self.participation_counts[client_id]
            current_part_num = self.currently_part_in[client_id]
            w = client.train_participation_normalised(part_cnt, current_part_num, copy.deepcopy(w_global)) 
            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
            self.currently_part_in[client_id] += 1
        return w_locals        

    def _train_clients_for_round(self, round_idx, w_global):
        w_locals = []
        client_indexes = self.sampling_functions[self.args.sampling_fun](
            round_idx, self.args.client_num_in_total
        )
        logging.info("client_indexes = %s", str(client_indexes))
        
        for idx, client in enumerate(self.client_list):
            if self.args.active and idx not in client_indexes:
                w = client.train()
                continue

            if not self.args.active:
                client_idx = client_indexes[idx]
                client.update_client(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx]
                )

            if self.args.active:
                current_model_params = client.model_trainer.get_model_params()
                last_aggregated_model_params = client.model_trainer.get_last_aggregated_model_params()
                globalVSlastAgg = self.sub_ordered_dict(w_global, last_aggregated_model_params)
                updated_state_dict = self.add_ordered_dict(self.args.alpha_active, current_model_params, globalVSlastAgg)  
                client.model_trainer.set_model_params(updated_state_dict)

            mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
            w = client.train(copy.deepcopy(w_global))
            # w = client.train()
            if self.args.active:
                client.model_trainer.set_last_aggregated_model_params(copy.deepcopy(w))
            mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))

            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
        return w_locals
            
    def _test_models_based_on_conditions(self, round_idx, w_global):
        if round_idx == self.args.comm_round - 1 or round_idx % self.args.frequency_of_the_test == 0:
            self._local_test_on_participating_clients(round_idx)
            self._test_global_model_on_global_data(w_global, round_idx)
            if self.args.group_wise_models:
                self._test_group_wise_model_on_local_data(round_idx)                    
    
    def _test_global_model_on_global_data(self, w_global, round_idx, return_val=False):
        logging.info("################test_global_model_on_global_dataset################")
        self.model_trainer.set_model_params(w_global)
        metrics_test = self.model_trainer.test(self.test_global, self.device, self.args)
        if return_val:
            return metrics_test
        # metrics_train = self.model_trainer.test(self.train_global, self.device, self.args)
        test_acc = metrics_test["test_correct"] / metrics_test["test_total"]
        test_loss = metrics_test["test_loss"] / metrics_test["test_total"]
        # train_acc = metrics_train["test_correct"] / metrics_train["test_total"]
        # train_loss = metrics_train["test_loss"] / metrics_train["test_total"]
        stats = {"test_acc": test_acc, "test_loss":test_loss}
        logging.info(stats)
        if self.args.enable_wandb:
            wandb.log({"Global Test Acc": test_acc}, step=round_idx)
            wandb.log({"Global Test Loss": test_loss}, step=round_idx)
            # wandb.log({"Global Train Acc": train_acc}, step=round_idx)
            # wandb.log({"Global Train Loss": train_loss}, step=round_idx)
            wandb.log({"Comm Round": round_idx}, step=round_idx)    
        
    def _local_test_on_participating_clients(self, round_idx, return_val=False):

        logging.info("################local_test_on_participating_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        # Define a nested function to handle the metrics collection to avoid redundancy
        def collect_metrics(client, dataset_type):
            metrics = client.local_test(dataset_type)
            return metrics["test_total"], metrics["test_correct"], metrics["test_loss"]

        clients_to_test = self.client_list

        for idx, client in enumerate(clients_to_test):
            if not self.args.active and self.test_data_local_dict[idx] is None:
                continue

            train_samples, train_correct, train_loss = collect_metrics(client, False)
            test_samples, test_correct, test_loss = collect_metrics(client, True)

            train_metrics["num_samples"].append(train_samples)
            train_metrics["num_correct"].append(train_correct)
            train_metrics["losses"].append(train_loss)

            test_metrics["num_samples"].append(test_samples)
            test_metrics["num_correct"].append(test_correct)
            test_metrics["losses"].append(test_loss)

        if return_val:
            return (train_metrics, test_metrics)
        # Compute aggregated metrics
        def compute_aggregated_metrics(metrics):
            return sum(metrics["num_correct"]) / sum(metrics["num_samples"]), sum(metrics["losses"]) / sum(metrics["num_samples"])

        train_acc, train_loss = compute_aggregated_metrics(train_metrics)
        test_acc, test_loss = compute_aggregated_metrics(test_metrics)

        # Log metrics
        def log_metrics(prefix, acc, loss):
            if self.args.enable_wandb:
                wandb.log({f"{prefix}/Acc": acc, "round": round_idx}, step=round_idx)
                wandb.log({f"{prefix}/Loss": loss, "round": round_idx}, step=round_idx)
            mlops.log({f"{prefix}/Acc": acc, "round": round_idx})
            mlops.log({f"{prefix}/Loss": loss, "round": round_idx})
            logging.info({f"{prefix}_acc": acc, f"{prefix}_loss": loss})

        log_metrics("Train", train_acc, train_loss)
        log_metrics("Test", test_acc, test_loss)

    def _test_group_wise_model_on_local_data(self, round_idx, return_val=False):
        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        def collect_metrics(client, dataset_type):
            metrics = client.local_test(dataset_type)
            return metrics["test_total"], metrics["test_correct"], metrics["test_loss"]

        clients_to_test = self.client_list

        for client in clients_to_test:
            group_id = client.group_id
            group_models = [self.group_specific_models[i] for i in group_id]

            # Initialize the dictionary to store the averaged parameters
            avg_params = {k: torch.zeros_like(v, dtype=torch.float) for k, v in group_models[0].state_dict().items()}

            # Sum the parameters of each model
            for model in group_models:
                model_params = model.state_dict()
                for k, v in model_params.items():
                    avg_params[k] += v

            # Divide each parameter by the number of models to get the average
            for k in avg_params.keys():
                avg_params[k] /= len(group_models)

            # Set the averaged parameters to the client's model trainer
            client.model_trainer.set_model_params(avg_params)

            train_samples, train_correct, train_loss = collect_metrics(client, False)
            test_samples, test_correct, test_loss = collect_metrics(client, True)

            train_metrics["num_samples"].append(train_samples)
            train_metrics["num_correct"].append(train_correct)
            train_metrics["losses"].append(train_loss)

            test_metrics["num_samples"].append(test_samples)
            test_metrics["num_correct"].append(test_correct)
            test_metrics["losses"].append(test_loss)

        if return_val:
            return (train_metrics, test_metrics)
        # Calculate and log the averaged metrics using wandb
        avg_train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        avg_test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        avg_train_loss = sum(train_metrics["losses"]) / len(train_metrics["losses"])
        avg_test_loss = sum(test_metrics["losses"]) / len(test_metrics["losses"])

        wandb.log({
            f"Average Group Train Accuracy": avg_train_acc,
            f"Average Group Test Accuracy": avg_test_acc,
            f"Average Group Train Loss": avg_train_loss,
            f"Average Group Test Loss": avg_test_loss
        }, step = round_idx)    
    
    
    def _test_cycle_wise(self, cycle_wise_metrics, cycle_idx):
        # Calculate and log the cycle-wise average metrics
        for metric_type in ["train", "test"]:
            for model_type in ["local", "group", "global"]:
                if metric_type == "train" and model_type == "global":
                    continue
                prefix = f"cycle_wise_{metric_type}_metrics_{model_type}"
                num_samples = sum(cycle_wise_metrics[prefix]["num_samples"])
                num_correct = sum(cycle_wise_metrics[prefix]["num_correct"])
                total_loss = sum(cycle_wise_metrics[prefix]["losses"])

                avg_acc = num_correct / num_samples
                avg_loss = total_loss / len(cycle_wise_metrics[prefix]["losses"])
                
                wandb.log({f"{model_type}_{metric_type}_Acc": avg_acc, "cycle": cycle_idx})
                wandb.log({f"{model_type}_{metric_type}_Loss": avg_loss, "cycle": cycle_idx})
        
