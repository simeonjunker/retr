# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import math
import sys
import tqdm

from collections import defaultdict
from os.path import dirname, abspath, join


from models import utils
from eval_utils import decode

file_path = dirname(abspath(__file__))
module_path = join(file_path, 'nlgeval')
sys.path.append(module_path)
from nlgeval import NLGEval


def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            validation_loss += loss.item()

            pbar.update(1)
        
    return validation_loss / total


def normalize_with_tokenizer(sent, tokenizer): 
    """
    use tokenizer to normalize annotated captions 
    (corresponding to system output)
    """
    return tokenizer.decode(tokenizer.encode(sent), skip_special_tokens=True)


def eval_model(model, dataloader, tokenizer, 
               config, start_token, metrics_to_omit=[]): 
    """
    iterate through val_loader and calculate CIDEr scores for model
    (only works with batch_size=1 for now)
    """
    
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, 
        metrics_to_omit=metrics_to_omit
    )

    # construct reference dict
    references = defaultdict(list)
    for a in dataloader.dataset.annot_select:
        references[a[0]].append(a[2])
        
    hyps = []
    refs = []

    # decode imgs in val set
    for i, (img_id, image, masks, caps, cap_masks) in enumerate(tqdm(data_loader)):
            
        h = greedy(model, image, tokenizer, start_token, max_pos_embeddings=config.max_position_embeddings)
        hyps += [h]
        
        img_refs = references[img_id.item()]
        normalized_refs = [normalize_with_tokenizer(r, tokenizer) for r in img_refs]
        refs += [normalized_refs]
        
    # transpose references to get correct format
    transposed_refs = list(map(list, zip(*refs)))
    
    # calculate cider score from hypotheses and references
    metrics_dict = nlgeval.compute_metrics(
        ref_list=transposed_refs, hyp_list=hyps)
    
    return metrics_dict