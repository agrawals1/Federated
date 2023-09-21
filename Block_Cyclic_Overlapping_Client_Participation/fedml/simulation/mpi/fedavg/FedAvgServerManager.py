import logging

from .message_define import MyMessage
from .utils import transform_tensor_to_list
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedAVGServerManager(FedMLCommManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.client_indexes = []

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling_cyclic(
            self.args.round_idx,
            self.args.client_num_in_total
        )
        self.client_indexes = client_indexes 
        global_model_params = self.aggregator.get_global_model_params() 
        if self.args.active:            
            chosen_clients_vec = [1 if i in client_indexes else 0 for i in range(self.args.client_num_in_total)]
            for client in range(self.args.client_num_in_total):
                process_id = client + 1
                self.send_message_init_config(
                    process_id, global_model_params, client, chosen_clients_vec[client]
                )
        else:
            
            for client_index in client_indexes:
                receiver_id = client_index + 1
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, client_index, 1
                )


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            self.client_indexes.index(sender_id-1), model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            self.aggregator.test_on_server_for_all_clients(self.args.round_idx)

            # start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                self.finish()
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.args.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.args.round_idx]
            else:
                client_indexes = self.aggregator.client_sampling_cyclic(
                    self.args.round_idx,
                    self.args.client_num_in_total,            
                )
            self.client_indexes = client_indexes
            global_model_params = self.aggregator.get_global_model_params()
            if self.args.active:                                
                chosen_clients_vec = [1 if i in client_indexes else 0 for i in range(self.args.client_num_in_total)]
                for client in range(self.args.client_num_in_total):
                    process_id = client + 1
                    self.send_message_sync_model_to_client(
                        process_id, global_model_params, client, chosen_clients_vec[client]
                    )
            else:            
                for client_index in client_indexes:
                    receiver_id = client_index + 1
                    self.send_message_sync_model_to_client(
                        receiver_id, global_model_params, client_index, 1
                    )


    def send_message_init_config(self, receive_id, global_model_params, client_index, aggregate):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params("participate_in_aggregation", aggregate)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, global_model_params, client_index, aggregate
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        if aggregate:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        else:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS,None)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
