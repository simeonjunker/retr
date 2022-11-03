import torch
from torch.utils.data import DataLoader
import argparse
from models import caption
from datasets import coco
from configuration import Config
import os
from tqdm import tqdm

from eval_utils.decode import prepare_tokenizer, load_image, greedy


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
    if checkpoint_path is None:
        raise NotImplementedError('No model to chose from!')
    else:
        if not os.path.exists(checkpoint_path):
            raise NotImplementedError('Give valid checkpoint path')

    print("Found checkpoint! Loading!")
    model, _ = caption.build_model(config)

    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model']) 

    return model   


def main_image(args, config):

    assert args.path is not None
    image_path = args.path

    # model
    model = prepare_model(args, config)

    # tokenizer
    tokenizer, start_token, end_token = prepare_tokenizer()

    # image handling
    image = load_image(image_path, transform=coco.val_transform)

    # decoding
    output = greedy(model, image, tokenizer, start_token,
                    config.max_position_embeddings)

    return output


def main_val_set(args, config):

    # model
    model = prepare_model(args, config)

    # tokenizer
    tokenizer, start_token, end_token = prepare_tokenizer()

    output = evaluate_val_set(
        config, model, tokenizer, start_token, config.max_position_embeddings
    )

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')

    parser.add_argument('--path', type=str,
                        help='path to image', default=None)
    parser.add_argument('--checkpoint', type=str,
                        help='checkpoint path', default='./checkpoint.pth')
    args = parser.parse_args()

    config = Config()

    print(main_val_set(args, config))
