import sys
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
sys.path.append(str(Path(__file__).parent.parent.resolve()))
import fedml
from fedml import FedMLRunner

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
