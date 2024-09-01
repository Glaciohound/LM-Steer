from pprint import pprint
import argparse
from .utils import set_seed


def parse_args():
    # Model related
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default='EleutherAI/gpt-neo-2.7B')
    parser.add_argument("--adaptor_class", type=str, default="multiply")
    parser.add_argument("--adapted_component", type=str, default="final_layer")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--init_var", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=1000)
    parser.add_argument("--num_steers", type=int, default=10)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--low_resource_mode", action="store_true")

    # Data related
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)

    # Training related
    parser.add_argument("--regularization", type=float, default=0)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma_mean", type=float, default=0.99)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_step", type=int, default=500)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--dummy_steer", type=int, default=None)
    parser.add_argument("--training_steer", type=int, default=0)

    # Evaluation related
    parser.add_argument("--eval_size", type=int, default=None)
    parser.add_argument("--steer_values", default=None, nargs="*", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--top_p", type=float, default=1)

    # transfer related
    parser.add_argument("--transfer_from", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    print("arguments:")
    pprint(args.__dict__)
    return args
