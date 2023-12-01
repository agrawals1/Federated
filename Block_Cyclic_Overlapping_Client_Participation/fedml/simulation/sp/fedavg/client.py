import math
class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global = None):
        # self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights
    
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

        # Train the model on the selected data slice
        self.model_trainer.train(train_data_slice, self.device, self.args)

        # Retrieve and return the updated model weights
        weights = self.model_trainer.get_model_params()
        return weights


    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
