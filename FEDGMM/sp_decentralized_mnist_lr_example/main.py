import fedml
from fedml import FedMLRunner
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
current_file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_dir)
if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
