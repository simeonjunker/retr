import copy
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn
import numpy as np

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def make_3d_att_mask(target_mask, source_mask, num_heads):

    # get dimensions per mask
    d_t = target_mask.shape[-1]
    d_s = source_mask.shape[-1]
    
    # repeat to get 2D masks
    exp_q = target_mask.unsqueeze(1).repeat(1,d_s,1)  # [b, d_t] -> [b, d_t, d_s]
    exp_k = source_mask.unsqueeze(1).repeat(1,d_t,1)  # [b, d_s] -> [b, d_s, d_t]
    
    # transpose q
    exp_q = torch.transpose(exp_q, 2, 1)  # [b, d_t, d_s] -> [b, d_s, d_t]
    
    # merge masks
    attn_mask_2d = torch.logical_or(
        exp_q,
        exp_k
    )
    
    attn_mask_3d = torch.repeat_interleave(attn_mask_2d, num_heads, dim=0)  # [N * H, T, S]
    
    return attn_mask_3d


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def ensure_unmasked_values(mask, unmasked_ratio = 0.01):
    """
    if mask is entirely True (i.e. everything is completely masked out):
    set some of the mask values to True
    (otherwise this leads to NaN values if model attends to inputs which are completely masked out)

    Args:
        mask (tensor): Mask tensor
        unmasked_ratio (float, optional): Ratio of unmasked values. Defaults to 0.01.
    """
    
    b, h, w = mask.shape
    flatten_mask = mask.reshape((b, -1))

    # check if all items contain unmasked values
    dim_contains_unmasked = torch.any(flatten_mask == False, dim=1)
    
    if False in dim_contains_unmasked:

        filler_mask = torch.ones((h, w), dtype=bool, device=mask.device.type)  # create dummy mask filled with True
        flatten_filler_mask = filler_mask.flatten()

        flatten_pixel_idx = np.indices(flatten_filler_mask.shape).flatten()  # idx of flattened mask
        idx_sample = np.random.choice(flatten_pixel_idx, replace=False, size=round(flatten_pixel_idx.size * unmasked_ratio))  # sample from flattened idx
        flatten_filler_mask[idx_sample] = False  # set sample values to False (i.e. do not mask out)
        
        filler_mask = flatten_filler_mask.reshape((h,w))  # reshape back to original format
        mask[~dim_contains_unmasked] = filler_mask  # replace mask with only True values with filler mask
    
    return mask


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        self.shape = self.mask.shape

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
