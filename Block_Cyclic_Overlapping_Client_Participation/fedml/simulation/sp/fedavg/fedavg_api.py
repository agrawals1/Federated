import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from collections import OrderedDict
from ClientSampler import ClientSampler
import copy

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
        self.sampling_functions = {"_client_sampling": self.client_sampler._client_sampling, 
                              "client_sampling_cyclic_overlap_pattern": self.client_sampler.client_sampling_cyclic_overlap_pattern,
                              "client_sampling_cyclic_noOverlap_random": self.client_sampler.client_sampling_cyclic_noOverlap_random,
                              "client_sampling_cyclic_overlap_random": self.client_sampler.client_sampling_cyclic_overlap_random,
                              "all_participate": self.client_sampler.all_participate
                              }
        
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model,copy.deepcopy(model),args)
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, copy.deepcopy(self.model_trainer),
        )

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
                )
                self.client_list.append(c)
        else:
            for client_idx in range(self.args.client_num_per_round):
                c = Client(
                    client_idx,
                    train_data_local_dict[client_idx],
                    test_data_local_dict[client_idx],
                    train_data_local_num_dict[client_idx],
                    self.args,
                    self.device,
                    model_trainer,
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
                client.update_local_dataset(
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
            # w = client.train(copy.deepcopy(w_global))
            w = client.train()
            if self.args.active:
                client.model_trainer.set_last_aggregated_model_params(copy.deepcopy(w))
            mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))

            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
        return w_locals

    # def _test_models_based_on_conditions(self, round_idx, w_global):
    #     if round_idx == self.args.comm_round - 1:
    #         self._local_test_on_all_clients(round_idx)
    #         self._test_global_model_on_global_data(w_global, round_idx)
    #     elif round_idx % self.args.frequency_of_the_test == 0:
    #         if self.args.dataset.startswith("stackoverflow"):
    #             self._local_test_on_validation_set(round_idx)
    #         else:
    #             self._local_test_on_all_clients(round_idx)
    #             self._test_global_model_on_global_data(w_global, round_idx)
    
    def _test_models_based_on_conditions(self, round_idx, w_global):
        if round_idx == self.args.comm_round - 1 or round_idx % self.args.frequency_of_the_test == 0:
            self._local_test_on_participating_clients(round_idx)
            self._test_global_model_on_global_data(w_global, round_idx)
    
    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

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

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _test_global_model_on_global_data(self, w_global, round_idx):
        logging.info("################test_global_model_on_global_dataset################")
        self.model_trainer.set_model_params(w_global)
        metrics_test = self.model_trainer.test(self.test_global, self.device, self.args)
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

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        # Define a nested function to handle the metrics collection to avoid redundancy
        def collect_metrics(client, dataset_type):
            metrics = client.local_test(dataset_type)
            return metrics["test_total"], metrics["test_correct"], metrics["test_loss"]

        if self.args.active:
            clients_to_test = self.client_list
        else:
            clients_to_test = [self.client_list[0] for _ in range(self.args.client_num_in_total)]

        for idx, client in enumerate(clients_to_test):
            if not self.args.active and self.test_data_local_dict[idx] is None:
                continue

            if not self.args.active:
                client.update_local_dataset(
                    0,
                    self.train_data_local_dict[idx],
                    self.test_data_local_dict[idx],
                    self.train_data_local_num_dict[idx],
                )

            train_samples, train_correct, train_loss = collect_metrics(client, False)
            test_samples, test_correct, test_loss = collect_metrics(client, True)

            train_metrics["num_samples"].append(train_samples)
            train_metrics["num_correct"].append(train_correct)
            train_metrics["losses"].append(train_loss)

            test_metrics["num_samples"].append(test_samples)
            test_metrics["num_correct"].append(test_correct)
            test_metrics["losses"].append(test_loss)

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
        
    def _local_test_on_participating_clients(self, round_idx):

        logging.info("################local_test_on_participating_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        # Define a nested function to handle the metrics collection to avoid redundancy
        def collect_metrics(client, dataset_type):
            metrics = client.local_test(dataset_type)
            return metrics["test_total"], metrics["test_correct"], metrics["test_loss"]

        if self.args.active:
            clients_to_test = self.client_list  ####################### Check this Logic #########################################
        else:
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



    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx}, step=round_idx)
                wandb.log({"Test/Loss": test_loss, "round": round_idx}, step=round_idx)

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx}, step=round_idx)
                wandb.log({"Test/Pre": test_pre, "round": round_idx}, step=round_idx)
                wandb.log({"Test/Rec": test_rec, "round": round_idx}, step=round_idx)
                wandb.log({"Test/Loss": test_loss, "round": round_idx}, step=round_idx)

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
