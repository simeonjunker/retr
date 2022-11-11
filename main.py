import torch
from torch.utils.data import DataLoader

import numpy as np
import time
import sys
import os

from models import utils, caption
from datasets import coco
from configuration import Config
from engine import train_one_epoch, evaluate
from train_utils.checkpoints import load_ckp, save_ckp, get_latest_checkpoint


def main(config):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, criterion = caption.build_model(config)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    dataset_train = coco.build_dataset(config, mode='training')
    dataset_val = coco.build_dataset(config, mode='validation')
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

    if not os.path.exists(config.checkpoint_path):
        os.mkdir(config.checkpoint_path)
    cpt_template = f'{config.prefix}_checkpoint_#.pth'

    if config.resume_training:
        # load latest checkpoint available
        latest_checkpoint = get_latest_checkpoint(config)
        if latest_checkpoint is not None:
            print(f'loading checkpoint: {latest_checkpoint}')
            epoch, model, optimizer, lr_scheduler, _, _ = load_ckp(
                model, optimizer, lr_scheduler, 
                path=os.path.join(config.checkpoint_path, latest_checkpoint)
            )
            config.start_epoch = epoch + 1
        else: 
            print(f'no suitable checkpoints found in {config.checkpoint_path}, starting training from scratch!')

    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        checkpoint_name = cpt_template.replace('#', str(epoch))
        save_ckp(
            epoch, model, optimizer, lr_scheduler, 
            train_loss=epoch_loss, val_loss=validation_loss, 
            path=os.path.join(config.checkpoint_path, checkpoint_name)
        )

        print()


if __name__ == "__main__":
    config = Config()
    main(config)
