import torch
from torch.utils.data import DataLoader
import argparse
from models import caption
from data_utils import refcoco
from configuration import Config
import os
import json
import re

from eval_utils.decode import prepare_tokenizer
from engine import eval_model


def prepare_model(args, config):

    # load model
    assert args.checkpoint is not None
        
    noise = noise_from_checkpoint(args.checkpoint)

    if not os.path.exists(args.checkpoint):
        raise NotImplementedError("Give valid checkpoint path")
    else:
        model, _ = caption.build_model(config)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    return model, noise


def setup_val_dataloader(config, noise_coverage, split='validation'):
    dataset_val = refcoco.build_dataset(
        config, 
        mode=split, 
        noise_coverage=noise_coverage,
        return_unique=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        sampler=sampler_val,
        drop_last=False,
        num_workers=config.num_workers,
    )
    return data_loader_val


def noise_from_checkpoint(checkpoint):
    match = re.search(r'noise(\d\-\d+)', checkpoint)
    if match:
        return float(match.group(1).replace('-', '.'))


def main_val_set(args, config):

    # model
    model, noise = prepare_model(args, config)
    model.to(args.device)
    print(f'Successfully loaded {model.__class__.__name__} model, noise = {noise}')

    # tokenizer
    tokenizer, _, _ = prepare_tokenizer()

    data_loader = setup_val_dataloader(config, noise_coverage=noise, split=args.split)

    metrics, generated = eval_model(
        model, data_loader, tokenizer, config, print_samples=args.print_samples
    )

    return metrics, generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REG")

    parser.add_argument(
        "--split", type=str.lower, choices=["val", "testa", "testb"], default="val"
    )
    parser.add_argument("--path", type=str, help="path to image", default=None)
    parser.add_argument("--checkpoint", type=str, help="checkpoint path", default=None)
    parser.add_argument(
        "--device", type=str.lower, choices=["cuda", "cpu", "auto"], default="auto"
    )
    parser.add_argument("--print_samples", action="store_true")
    parser.add_argument("--store_results", action="store_true")
    parser.add_argument("--override_config", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config()

    metrics, generated = main_val_set(args, config)

    print(metrics)

    if args.store_results:
        assert args.checkpoint is not None
        model_name = os.path.split(args.checkpoint)[-1]
        outdir = os.path.abspath("./data/results")
        if not os.path.isdir(outdir):
            print(f"create output directory {outdir}")
            os.makedirs(outdir)
        # generated expressions
        outfile_name = model_name.replace(".pth", f"_{args.split}_generated.json")
        outfile_path = os.path.join(outdir, outfile_name)
        print(f"write generated expressions to {outfile_path}")
        with open(outfile_path, "w") as f:
            json.dump(generated, f)
        # evaluation results
        outfile_name = model_name.replace(".pth", f"_{args.split}_eval.json")
        outfile_path = os.path.join(outdir, outfile_name)
        print(f"write evaluation results to {outfile_path}")
        with open(outfile_path, "w") as f:
            json.dump(metrics, f)