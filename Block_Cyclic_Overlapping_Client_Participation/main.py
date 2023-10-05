import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
import fedml
from fedml import FedMLRunner
from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist
import wandb
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent

def wandB_init(args):
    if hasattr(args, "enable_wandb") and args.enable_wandb:
        wandb_only_server = getattr(args, "wandb_only_server", None)
        if (wandb_only_server and args.rank == 0 and args.process_id == 0) or not wandb_only_server:
            wandb_entity = getattr(args, "wandb_entity", None)
            if wandb_entity is not None:
                wandb_args = {
                    "entity": args.wandb_entity,
                    "project": args.wandb_project,
                    "config": args,
                }
            else:
                wandb_args = {
                    "project": args.wandb_project,
                    "config": args,
                }

            if hasattr(args, "run_name"):
                wandb_args["name"] = args.run_name

            if hasattr(args, "wandb_group_id"):
                # wandb_args["group"] = args.wandb_group_id
                wandb_args["group"] = "Test1"
                wandb_args["name"] = f"Client {args.rank}"
                wandb_args["job_type"] = str(args.rank)     

            wandb.init(**wandb_args)
            MLOpsProfilerEvent.enable_wandb_tracking()

if __name__ == "__main__":
    # init FedML framework
    runs = ["random_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet",
            "cyclic_NoOverlap_Pattern_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet",
            "cyclic_Overlap_Pattern_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet",
            "cyclic_NoOverlap_Random_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet",
            "cyclic_Overlap_Random_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet",     
            ]
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args) 

    # load data
    dataset, output_dim = fedml.data.load(args)

    for run in runs:
        args.run_name = run
        if run == "random_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet":
            args.sampling_func = "_client_sampling"
        elif run == "cyclic_NoOverlap_Pattern_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet":
            args.sampling_func = "client_sampling_cyclic_noOverlap_pattern"
        elif run == "cyclic_Overlap_Pattern_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet":
            args.sampling_func = "client_sampling_cyclic_overlap_pattern"
        elif run == "cyclic_NoOverlap_Random_E4_R50_LRe-1_DEC5e-4_COS_rsnt18_cfar10_Dirichlet":
            args.sampling_func = "client_sampling_cyclic_noOverlap_random"
        else:
            args.sampling_func = "client_sampling_cyclic_overlap_random"
            
        # Create run
        wandB_init(args)

        # load model
        model = fedml.model.create(args, output_dim)

        # start training
        fedml_runner = FedMLRunner(args, device, dataset, model)
        fedml_runner.run()
        wandb.finish()