import torch
from torch.utils.data import DataLoader
import argparse
from models import caption
from datasets import coco
from configuration import Config
import os
from tqdm import tqdm

from eval_utils.decode import prepare_tokenizer, load_image, greedy
from train_utils.checkpoints import get_latest_checkpoint
from engine import eval_model

def setup_val_dataloader(config):
    dataset_val = coco.build_dataset(config, mode='validation', return_unique=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 
                                 batch_size=1,
                                 sampler=sampler_val, drop_last=False,
                                 num_workers=config.num_workers)
    return data_loader_val


def evaluate_val_set(
        config, model, tokenizer,
        start_token, max_pos_embeddings
        ):
    data_loader = setup_val_dataloader(config)

    all_caps = []

    for image, masks, caps, cap_masks in tqdm(data_loader):
        c = greedy(model, image, tokenizer, start_token, max_pos_embeddings)
        all_caps.append(c)
        print(c)

    return all_caps


def prepare_model(args, config):

    checkpoint_path = args.checkpoint

    # load model
    if args.checkpoint is not None:
      if not os.path.exists(args.checkpoint):
        raise NotImplementedError('Give valid checkpoint path')
      else:
        model,_ = caption.build_model(config)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
      print("Checking for checkpoint.")
      checkpoint_path = config.checkpoint_path
      if checkpoint_path is None:
        raise NotImplementedError('No checkpoint path given!')
      elif not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      else:
        latest_cpt = get_latest_checkpoint(config)
        if latest_cpt is not None:      
          print("Found checkpoint! Loading!")
          model,_ = caption.build_model(config)
          checkpoint = torch.load(os.path.join(checkpoint_path, latest_cpt), map_location='cpu')
          model.load_state_dict(checkpoint['model_state_dict'])
        else:
          print(f"No valid checkpoint found in {checkpoint_path}")

    return model  


def setup_val_dataloader(config):
    dataset_val = coco.build_dataset(config, mode='validation', return_unique=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 
                                 batch_size=config.batch_size,
                                 sampler=sampler_val, drop_last=False,
                                 num_workers=config.num_workers)
    return data_loader_val


def main_image(args, config):

    assert args.path is not None
    image_path = args.path

    # model
    model = prepare_model(args, config)

    # tokenizer
    tokenizer, start_token, end_token = prepare_tokenizer()
    bos_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    # image handling
    image = load_image(image_path, transform=coco.val_transform)

    # decoding
    output_ids = greedy(
      image, model, 
      max_len=config.max_position_embeddings, device="auto", 
      bos_token=bos_id, eos_token=eos_id)

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output


def main_val_set(args, config):

    # model
    model = prepare_model(args, config)

    # tokenizer
    tokenizer, start_token, end_token = prepare_tokenizer()

    data_loader = setup_val_dataloader(config)

    metrics = eval_model(model, data_loader, tokenizer, config)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')

    parser.add_argument('--path', type=str,
                        help='path to image', default=None)
    parser.add_argument('--checkpoint', type=str,
                        help='checkpoint path', default=None)
    parser.add_argument('--mode', default='val')
    args = parser.parse_args()

    config = Config()

    if args.mode == 'val':
        print(main_val_set(args, config))
    elif args.mode == 'image':
        print(main_image(args, config))
