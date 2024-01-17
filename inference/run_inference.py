import sys
import os
from os.path import dirname, abspath, join, split
import torch
import argparse
import json
import pickle
from inference_utils import main_val_set_with_att

file_path = abspath(dirname(__file__))
sys.path.append(abspath(join(file_path, os.pardir, 'model')))

from configuration import Config

def main(args, model_args, config):

        # decode dataset
        metrics, generated = main_val_set_with_att(args, model_args, config)

        print(metrics)

        assert args.checkpoint is not None
        model_name = split(args.checkpoint)[-1]
        
        if args.auto_checkpoint_path:
            
            noise_str = str(model_args.target_noise).replace(".", "-")
            
            if config.use_global_features:
                context_str = 'context'
            elif config.use_scene_summaries:
                context_str = 'scene'
            else:
                context_str = 'nocontext'
                
            outdir = os.path.join(args.output_directory, 'models', config.prefix, f'noise_{noise_str}_{context_str}')
            
        else: 
            outdir = args.output_directory
        
        if not os.path.isdir(outdir):
            print(f"create output directory {outdir}")
            os.makedirs(outdir)

        # generated expressions
        outfile_name = model_name.replace(".pth", f"_{args.split}_generated.pkl")
        outfile_path = os.path.join(outdir, outfile_name)
        print(f"write generated expressions to {outfile_path}")
        with open(outfile_path, "wb") as f:
            pickle.dump(generated, f)

        # evaluation results
        outfile_name = model_name.replace(".pth", f"_{args.split}_eval.json")
        outfile_path = os.path.join(outdir, outfile_name)
        print(f"write evaluation results to {outfile_path}")
        with open(outfile_path, "w") as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REG")

    parser.add_argument(
        "--split", type=str.lower, choices=["val", "testa", "testb"], default="val"
    )
    parser.add_argument("--path", type=str, help="path to image", default=None)
    parser.add_argument("--checkpoint", type=str, help="checkpoint path", default=None, required=True)
    parser.add_argument(
        "--device", type=str.lower, choices=["cuda", "cpu", "auto"], default="auto"
    )
    parser.add_argument("--print_samples", action="store_true")
    parser.add_argument("--output_directory", default=os.path.abspath("../data/results"))
    parser.add_argument("--auto_checkpoint_path", default=True, type=bool)

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    local_config = Config()

    checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
    assert 'config' in checkpoint_data.keys()
    
    print('using config from checkpoint')
    config = checkpoint_data['config']
    model_args = checkpoint_data['args']
    config.dir = local_config.dir
    config.ref_base = local_config.ref_base
    config.ref_dir = local_config.ref_dir

    config.batch_size = 1  # override batch size

    print('config:', vars(config))
    print('args: ', vars(args))

    main(args, model_args, config)