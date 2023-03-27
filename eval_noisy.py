import torch
from torch.utils.data import DataLoader
import argparse
from models import caption
from datasets import refcoco
from configuration import Config
import os
import json
import numpy as np
import torchvision
from torchvision.transforms import Resize, Compose, ToTensor, Lambda, Normalize

from eval_utils.decode import prepare_tokenizer
from engine import eval_model


def prepare_model(args, config):

    # load model
    assert args.checkpoint is not None

    if args.override_config:
        # overriding config settings with parameters given by checkpoint
        override_config_with_checkpoint(args.checkpoint, config)

    if not os.path.exists(args.checkpoint):
        raise NotImplementedError("Give valid checkpoint path")
    else:
        model, _ = caption.build_model(config)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    return model


def cover_with_noise(image_tensor, coverage=0.5):
    # create 1D selection mask
    _, h, w = image_tensor.shape
    mask = np.zeros((h, w), dtype=bool).flatten()  # [hw]

    # sample from flattened index & set mask to True
    idx = np.indices(mask.shape).flatten()
    idx_sample = np.random.choice(idx, replace=False, size=round(mask.size * coverage))
    mask[idx_sample] = True
    # reshape to 2D image shape
    mask = mask.reshape((h, w))  # [h, w]

    # mask image with noise
    noise_tensor = torch.rand_like(image_tensor)
    image_tensor[:, mask] = noise_tensor[:, mask]

    return image_tensor


def get_transforms(config, noise_coverage):

    backbone_weights = getattr(torchvision.models, config.backbone + '_Weights').DEFAULT
    default_transforms = backbone_weights.transforms()

    resize = Resize(
        size=default_transforms.crop_size,
        interpolation=default_transforms.interpolation
    )

    target_transform = Compose([
            ToTensor(),
            Lambda(lambda x: cover_with_noise(x, noise_coverage)),
            Normalize(mean=default_transforms.mean, 
                      std=default_transforms.std),
        ])

    context_transform = Compose([
            ToTensor(),
            Normalize(mean=default_transforms.mean, 
                      std=default_transforms.std),
        ])

    return resize, target_transform, context_transform


def setup_val_dataloader(config, noise_coverage):
    resize, target_transform, context_transform = get_transforms(
        config, noise_coverage)

    transform = {
        'target': {
            'resize': resize,
            'transform': target_transform,
        },
        'context': {
            'resize': resize,
            'transform': context_transform,
        }
    }

    dataset_val = refcoco.build_dataset(
        config, 
        mode="validation", 
        transform=transform,
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


def override_config_with_checkpoint(checkpoint, config):
    use_glob = config.use_global_features
    use_loc = config.use_location_features
    
    if 'loc_checkpoint' in checkpoint:
        if not (not use_glob and use_loc):
            # override settings
            config.use_global_features = False
            config.use_location_features = True
            # send warning
            print(f'''CAUTION: Overriding configuration!
                WAS: use_global_features=={use_glob}; use_location_features=={use_loc}
                NEW: use_global_features=={config.use_global_features}; use_location_features=={config.use_location_features}
                ''')
            
    elif 'loc_glob_checkpoint' in checkpoint:
        if not (use_glob and use_loc):
            # override settings
            config.use_global_features = True
            config.use_location_features = True
            # send warning
            print(f'''CAUTION: Overriding configuration!
                WAS: use_global_features=={use_glob}; use_location_features=={use_loc}
                NEW: use_global_features=={config.use_global_features}; use_location_features=={config.use_location_features}
                ''')
            
    else:
        raise NotImplementedError(
            "Overriding model checkpoints is not supported for the model type given by the checkpoint"
        )


def main_val_set(args, config):

    # model
    model = prepare_model(args, config).to(args.device)

    # tokenizer
    tokenizer, _, _ = prepare_tokenizer()

    data_loader = setup_val_dataloader(config, args.noise_coverage)

    metrics, generated = eval_model(
        model, data_loader, tokenizer, config, print_samples=args.print_samples
    )

    return metrics, generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REG")

    parser.add_argument("--mode", default="val")
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
    parser.add_argument(
        "--noise_coverage", type=float, default=0.5, 
        help="proportion of the target image to be covered with random noise"
    )
    parser.add_argument("--override_config", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config()

    if args.mode == "val":

        metrics, generated = main_val_set(args, config)

        print(metrics)

        if args.store_results:
            assert args.checkpoint is not None
            model_name = os.path.split(args.checkpoint)[-1]
            outdir = os.path.abspath("./data/results")
            noise_str = str(args.noise_coverage).replace('.', '-')
            if not os.path.isdir(outdir):
                print(f"create output directory {outdir}")
                os.makedirs(outdir)
            # generated expressions
            outfile_name = model_name.replace(
                ".pth", f"_{args.split}_noise{noise_str}_generated.json")
            outfile_path = os.path.join(outdir, outfile_name)
            print(f"write generated expressions to {outfile_path}")
            with open(outfile_path, "w") as f:
                json.dump(generated, f)
            # evaluation results
            outfile_name = model_name.replace(
                ".pth", f"_{args.split}_noise{noise_str}_eval.json")
            outfile_path = os.path.join(outdir, outfile_name)
            print(f"write evaluation results to {outfile_path}")
            with open(outfile_path, "w") as f:
                json.dump(metrics, f)
