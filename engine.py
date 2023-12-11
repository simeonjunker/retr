# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import math
import sys
import tqdm

from collections import defaultdict
from os.path import dirname, abspath, join

from models.utils import NestedTensor
from eval_utils.decode import greedy_decoding

file_path = dirname(abspath(__file__))
module_path = join(file_path, 'nlgeval')
sys.path.append(module_path)
from nlgeval import NLGEval


def pack_encoder_inputs(encoder_input,
                        global_features,
                        location_features,
                        scene_features,
                        device='cpu'):

    if not global_features and not location_features and not scene_features:
        # default / target only
        t_img, t_mask = encoder_input
        # return as tuple w/ len 1
        return (NestedTensor(t_img, t_mask).to(device), )
    if global_features and not location_features and not scene_features:
        # target + global
        t_img, t_mask, g_img, g_mask = encoder_input
        # return as tuple w/ len 2
        return (NestedTensor(t_img, t_mask).to(device),
                NestedTensor(g_img, g_mask).to(device))
    elif not global_features and location_features and not scene_features:
        # target + location
        t_img, t_mask, l_feats = encoder_input
        # return as tuple w/ len 2
        return (NestedTensor(t_img, t_mask).to(device), l_feats.to(device))
    elif global_features and location_features and not scene_features:
        # target + global + location
        t_img, t_mask, g_img, g_mask, l_feats = encoder_input
        # return as tuple w/ len 3
        return (NestedTensor(t_img, t_mask).to(device),
                NestedTensor(g_img, g_mask).to(device), l_feats.to(device))
    elif not global_features and location_features and scene_features:
        # target + location + scene
        t_img, t_mask, l_feats, scene_feats = encoder_input
        # return as tuple w/ len 3
        return (NestedTensor(t_img, t_mask).to(device),
                l_feats.to(device),
                scene_feats.to(device))
    else:
        raise NotImplementedError()
        


def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    global_features = data_loader.dataset.return_global_context
    location_features = data_loader.dataset.return_location_features
    scene_features = data_loader.dataset.return_scene_features

    with tqdm.tqdm(total=total) as pbar:
        for i, (_, *encoder_input, caps, cap_masks) in enumerate(data_loader):          
            samples = pack_encoder_inputs(
                encoder_input, global_features, location_features, scene_features, device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(*samples, caps[:, :-1], cap_masks[:, :-1])
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

    global_features = data_loader.dataset.return_global_context
    location_features = data_loader.dataset.return_location_features
    scene_features = data_loader.dataset.return_scene_features


    with tqdm.tqdm(total=total) as pbar:
        for i, (_, *encoder_input, caps, cap_masks) in enumerate(data_loader):         
            samples = pack_encoder_inputs(
                encoder_input, global_features, location_features, scene_features, device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(*samples, caps[:, :-1], cap_masks[:, :-1])
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


def eval_model(model, data_loader, tokenizer,
               config, metrics_to_omit=[],
               print_samples=False):
    """
    iterate through val_loader and calculate CIDEr scores for model
    (only works with batch_size=1 for now)
    """

    model.eval()

    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True,
        metrics_to_omit=metrics_to_omit
    )

    # construct reference dict
    annotations = defaultdict(list)
    for a in data_loader.dataset.annot:
        annotations[a[0]].append(a[2])

    ids, hypotheses, ids_hypotheses, references = [], [], [], []

    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    bos_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    global_features = data_loader.dataset.return_global_context
    location_features = data_loader.dataset.return_location_features
    scene_features = data_loader.dataset.return_scene_features


    # decode imgs in val set
    for i, (ann_ids, *encoder_input, _, _) in enumerate(tqdm.tqdm(data_loader)):
        samples = pack_encoder_inputs(encoder_input, global_features, location_features, scene_features)

        # get model predictions
        hyps = greedy_decoding(
            samples, model, tokenizer,
            max_len=config.max_position_embeddings, clean=True,
            pad_token=pad_id, bos_token=bos_id, eos_token=eos_id,
            device='auto'
        )

        hypotheses += hyps

        ids_hyps = [{'ann_id': i, 'expression': h} for i,h in zip(ann_ids.tolist(), hyps)]
        ids_hypotheses += ids_hyps
        if print_samples:
            print(*ids_hyps, sep='\n')

        # get annotated references
        refs = [annotations[i.item()] for i in ann_ids]
        normalized_refs = [
            [normalize_with_tokenizer(r, tokenizer) for r in _refs] for _refs in refs
        ]
        references += normalized_refs

    # transpose references to get correct format
    transposed_references = list(map(list, zip(*references)))

    # calculate cider score from hypotheses and references
    metrics_dict = nlgeval.compute_metrics(
        ref_list=transposed_references, hyp_list=hypotheses)

    return metrics_dict, ids_hypotheses