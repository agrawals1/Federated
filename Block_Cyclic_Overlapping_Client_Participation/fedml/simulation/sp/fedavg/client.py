from copy import deepcopy
class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer, group_id
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.group_id = group_id
        if self.args.federated_optimizer == 'Scaffold':
            if self.args.data_split:
                self.c_models_local = {}
            else:
                self.c_model_local = deepcopy(self.model_trainer.model).cpu()
                for name, params in self.c_model_local.named_parameters():
                    params.data = params.data * 0

    def update_client(self, client_idx, local_training_data, local_test_data, local_sample_number, *args):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)
        if args:
            self.group_id = args[0]

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global = None):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights
    
    def train_Scaffold(self, c_model_global_param, w_global = None):
        c_model_global_param = deepcopy(c_model_global_param)
        c_model_local_param = self.c_model_local.state_dict()
        self.model_trainer.set_model_params(w_global)
        iteration_cnt = self.model_trainer.train(self.local_training_data, self.device, self.args, c_model_global_param, c_model_local_param)
        weights = self.model_trainer.get_model_params()

        c_new_para = self.c_model_local.cpu().state_dict()
        c_delta_para = {}
        weights_delta = {}
        for key in weights:
            c_new_para[key] = c_new_para[key] - c_model_global_param[key].cpu() + \
                            (w_global[key] - weights[key]) / (iteration_cnt * self.args.learning_rate)
            c_delta_para[key] = c_new_para[key] - c_model_local_param[key].cpu()
            weights_delta[key] = weights[key] - w_global[key].cpu()
        self.c_model_local.load_state_dict(c_new_para)
        return weights_delta, c_delta_para
    
    def train_participation_normalised(self, part_cnt, current_part_num, w_global=None):
        train_data = self.local_training_data

        # Calculate the number of batches per part
        total_batches = len(train_data)
        batches_per_part = total_batches // part_cnt

        # Determine the start and end index for the current part
        start_idx = batches_per_part * current_part_num
        end_idx = start_idx + batches_per_part if current_part_num < part_cnt - 1 else total_batches

        # Select the slice of training data for the current part
        train_data_slice = train_data[start_idx:end_idx]

        self.model_trainer.set_model_params(w_global)

        # Train the model on the selected data slice
        self.model_trainer.train(train_data_slice, self.device, self.args)

        # Retrieve and return the updated model weights
        weights = self.model_trainer.get_model_params()
        return weights
    
    def train_participation_normalised_Scaffold(self, part_cnt, current_part_num, c_model_global_param, w_global=None):
        if current_part_num not in self.c_models_local:
            self.c_models_local[current_part_num] = deepcopy(self.model_trainer.model)
            for name, param in self.c_models_local[current_part_num].named_parameters():
                param.data *= 0
        c_model_global_param = deepcopy(c_model_global_param)
        c_model_local_param = self.c_models_local[current_part_num].state_dict()
        train_data = self.local_training_data

        # Calculate the number of batches per part
        total_batches = len(train_data)
        batches_per_part = total_batches // part_cnt
        
        if part_cnt > total_batches:
             # Determine the start and end index for the current part
            start_idx = current_part_num % total_batches
            end_idx = start_idx + 1
        
        else:
            # Determine the start and end index for the current part
            start_idx = batches_per_part * current_part_num
            end_idx = start_idx + batches_per_part if current_part_num < part_cnt - 1 else total_batches

        # Select the slice of training data for the current part
        train_data_slice = train_data[start_idx:end_idx]
        self.model_trainer.set_model_params(w_global)
        # Train the model on the selected data slice
        iteration_cnt = self.model_trainer.train(train_data_slice, self.device, self.args, c_model_global_param, c_model_local_param)

        # Retrieve and return the updated model weights
        weights = self.model_trainer.get_model_params()
        c_new_para = self.c_models_local[current_part_num].state_dict()
        c_delta_para = {}
        weights_delta = {}
        for key in weights:
            c_new_para[key] = c_new_para[key] - c_model_global_param[key] + \
                            (w_global[key] - weights[key]) / (iteration_cnt * self.args.learning_rate)
            c_delta_para[key] = c_new_para[key] - c_model_local_param[key]
            weights_delta[key] = weights[key] - w_global[key]
        self.c_models_local[current_part_num].load_state_dict(c_new_para)
        return weights_delta, c_delta_para


    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
