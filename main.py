import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import logging
import json
import argparse
from glob import glob
import os.path as osp

from models import utils, caption
from data_utils import refcoco
from configuration import Config
from engine import train_one_epoch, evaluate, eval_model
from train_utils.checkpoints import save_ckp
from eval_utils.decode import prepare_tokenizer
from train_utils.early_stopping import ScoreTracker


def main(args, config):
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
    score_tracker = ScoreTracker(config.stop_after_epochs)
    tokenizer, _, _ = prepare_tokenizer()
    
    # build datasets
    if 'refcoco' in config.prefix:
        build_dataset = refcoco.build_dataset
    else:
        raise NotImplementedError

    dataset_train = build_dataset(
        config, mode='training', noise_coverage=args.target_noise)
    dataset_val = build_dataset(
        config, mode='validation', noise_coverage=args.target_noise)
    dataset_cider = build_dataset(
        config, mode='validation', return_unique=True, noise_coverage=args.target_noise)
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")
    print(f"CIDEr evaluation: {len(dataset_cider)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_cider = torch.utils.data.SequentialSampler(dataset_cider)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)
    data_loader_cider = DataLoader(dataset_cider, config.batch_size,
                                 sampler=sampler_cider, drop_last=False, num_workers=config.num_workers)

    noise_str = str(args.target_noise).replace(".", "-")

    if args.auto_checkpoint_path:
        
        if config.use_global_features:
            context_str = 'context'
        elif config.use_scene_summaries:
            context_str = 'scene'
        else:
            context_str = 'nocontext'
            
        checkpoint_path = os.path.join(config.project_data_path, 'models', config.prefix, f'noise_{noise_str}_{context_str}')
        
    else: 
        checkpoint_path = config.checkpoint_path
        
    print(f'Saving model checkpoints to {checkpoint_path}')
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
           
    loc_used = '_loc' if config.use_location_features else ''
    glob_used = '_glob' if config.use_global_features else ''
    scene_used = '_scene' if config.use_scene_summaries else ''
    model_name = f'{config.transformer_type}_{config.prefix}{loc_used}{glob_used}{scene_used}_noise{noise_str}'

    log_path = os.path.join(checkpoint_path, 'train_progress_' + model_name + '.log')
    logging.basicConfig(
        filename=log_path, 
        level=logging.DEBUG
    )

    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")
        logging.info(f'Train loss / epoch {epoch}: {epoch_loss}')

        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")
        logging.info(f'Val loss / epoch {epoch}: {validation_loss}')

        eval_results, ids_hypotheses = eval_model(model, data_loader_cider, tokenizer, config)
        cider_score = eval_results['CIDEr']
        print(f"CIDEr score: {cider_score}")
        logging.info(f'CIDEr score / epoch {epoch}: {cider_score}')
        
        if args.save_samples:
            sample_name = f"{model_name}-{epoch:03d}-samples.json"
            with open(
                os.path.join(checkpoint_path, sample_name),
                "w",
            ) as f:
                json.dump(ids_hypotheses, f)

        # early stopping / export model weights based on CIDEr score
        score_tracker(cider_score)
        score_tracker.print_summary()
        
        if isinstance(config.save_every, int):
            save_model = (epoch % config.save_every == 0 or epoch == config.epochs - 1)
        else:
            save_model = score_tracker.counter == 0
            if not save_model:
                print('non maximum score -- do not save model weights')

        if save_model:
            
            checkpoint_name = model_name + f'_checkpoint_{epoch:03d}.pth'
            print(f'save model weights to {checkpoint_name}')
            save_ckp(
                epoch, model, optimizer, lr_scheduler, args, config,
                train_loss=epoch_loss, val_loss=validation_loss, cider_score=cider_score,
                path=os.path.join(checkpoint_path, checkpoint_name)
            )
            
            if args.clean_old_checkpoints:
                for old_epoch in range(epoch):
                    old_checkpoint = model_name + f'_checkpoint_{old_epoch:03d}.pth'
                    old_path = os.path.join(checkpoint_path, old_checkpoint)
                    if os.path.isfile(old_path):
                        print(f'removing old checkpoint {old_checkpoint}')
                        os.remove(old_path)
                
        
        if score_tracker.stop():
            break

        print()


if __name__ == "__main__":
    config = Config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_noise", default=0.0, type=float)
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--use_scene_summaries", action='store_true')
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--auto_checkpoint_path", default=True, type=bool)
    parser.add_argument("--clean_old_checkpoints", default=True, type=bool)
    args = parser.parse_args()
    
    config.use_global_features = args.use_context
    config.use_scene_summaries = args.use_scene_summaries

    if args.dataset is not None:
        print(f'overwrite config dataset ({config.prefix}) with ({args.dataset}) from args')
        config.prefix = args.dataset
        config.ref_dir = osp.join(config.ref_base, config.prefix)
        config.checkpoint_path = osp.join(config.project_data_path, 'models', config.prefix)
    
    main(args, config)
