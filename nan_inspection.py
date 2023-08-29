import torch
from torch.utils.data import DataLoader

import numpy as np
import time
import sys
import os

from models import utils, caption
from data_utils import refcoco
from configuration import Config
from engine import train_one_epoch, evaluate, eval_model
from train_utils.checkpoints import load_ckp, save_ckp
from eval_utils.decode import prepare_tokenizer

from engine import pack_encoder_inputs
import math

config = Config()
#fail_idx = 116348
fail_idx = 42

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
tokenizer, _, _ = prepare_tokenizer()

print(f"Building {config.prefix} dataset, data root: {config.ref_dir}")
dataset_train = refcoco.build_dataset(config, mode='training')
print(f"Train: {len(dataset_train)}")

sampler_train = torch.utils.data.RandomSampler(dataset_train)


batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, config.batch_size, drop_last=True
)

data_loader_train = DataLoader(
    dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)



# forward pass with fail ids




i = fail_idx

ann_ids, *encoder_input, ns_encoder_inputs, caps, cap_masks = dataset_train[i]

encoder_input = [e.unsqueeze(0) for e in encoder_input]
caps = torch.from_numpy(caps).unsqueeze(0)
cap_masks = torch.from_numpy(cap_masks).unsqueeze(0)

samples = pack_encoder_inputs(
                encoder_input, dataset_train.return_global_context, dataset_train.return_location_features, dataset_train.return_scene_features, device)

outputs = model(*samples, caps[:, :-1], cap_masks[:, :-1])

loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

loss_value = loss.item()

if not math.isfinite(loss_value):
    print(f'Loss for {i} is {loss_value}, stopping training')
    #sys.exit(1)