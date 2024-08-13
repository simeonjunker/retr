import sys
import os
from os.path import dirname, abspath, join
import torch
import argparse
import json
import pickle
from inference_utils import main_val_set_with_att

file_path = abspath(dirname(__file__))
sys.path.append(abspath(join(file_path, os.pardir, 'model')))

from configuration import Config


def main(args, model_args, model_epoch, config):
    
    # parse info
    
    dataset_str = config.prefix

    architecture_str = 'trf'

    if model_args.use_context:
        context = 'global'
    elif model_args.use_scene_summaries:
        context = 'scene'
    else:
        context = 'nocontext'
    context_str = f'context:{context}'

    noise_str = f"noise:{str(model_args.target_noise).replace('.', '-')}"

    epoch_str = f"epoch:{str(model_epoch).rjust(2,'0')}"
    
    # create output dir
    
    if args.auto_checkpoint_path:
        outdir = os.path.join(args.output_directory, dataset_str, f'{noise_str.replace(":", "_")}_{context}')
    else: 
        outdir = args.output_directory
    
    if not os.path.isdir(outdir):
        print(f"create output directory {outdir}")
        os.makedirs(outdir)
    
    # decode dataset
    
    metrics, generated = main_val_set_with_att(args, model_args, config, skip_attentions=args.skip_attentions)
    print(metrics)

    # save generated expressions
    
    file_prefix = f'{dataset_str}_{args.split}_{architecture_str}_{context_str}_{noise_str}_{epoch_str}'
    idx_suffix = f'_{args.idx_suffix}' if args.idx_suffix is not None else ''

    outfile_name = f"{file_prefix}_generated{idx_suffix}.pkl"
    outfile_path = os.path.join(outdir, outfile_name)
    print(f"write generated expressions to {outfile_path}")
    with open(outfile_path, "wb") as f:
        pickle.dump(generated, f)

    # save evaluation results
    
    outfile_name = f"{file_prefix}_metrics{idx_suffix}.json"
    outfile_path = os.path.join(outdir, outfile_name)
    print(f"write evaluation results to {outfile_path}")
    with open(outfile_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REG")

    parser.add_argument(
        "--split", type=str.lower, choices=["val", "testa", "testb", "test"], default="val"
    )
    parser.add_argument("--path", type=str, help="path to image", default=None)
    parser.add_argument("--checkpoint", type=str, help="checkpoint path", default=None, required=True)
    parser.add_argument(
        "--device", type=str.lower, choices=["cuda", "cpu", "auto"], default="auto"
    )
    parser.add_argument("--print_samples", action="store_true")
    parser.add_argument("--output_directory", default=os.path.abspath("../data/results"))
    parser.add_argument("--auto_checkpoint_path", default=True, type=bool)
    parser.add_argument("--skip_attentions", action='store_true')
    parser.add_argument("--idx_suffix", default=None)

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    local_config = Config()

    checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
    assert 'config' in checkpoint_data.keys()
    
    print('using config from checkpoint')
    config = checkpoint_data['config']
    model_args = checkpoint_data['args']
    model_epoch = checkpoint_data['epoch']
    config.dir = local_config.dir
    config.project_data_path = local_config.project_data_path
    config.ref_base = local_config.ref_base
    config.ref_dir = join(config.ref_base, config.prefix)

    config.batch_size = 1  # override batch size

    print('config:', vars(config))
    print('args: ', vars(args))

    main(args, model_args, model_epoch, config)