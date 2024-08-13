import torch
from typing import Optional, List
from torch import Tensor
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
from math import floor, ceil
from decimal import Decimal

import json
import os

MAX_DIM = 224


def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = [3, MAX_DIM, MAX_DIM]
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
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


def get_refcoco_df(path):
    """get RefCOCO* annotations as pd.DataFrame

    Args:
        path (string): path to RefCOCO* base dir

    Returns:
        pd.DataFrame: RefCOCO* annotations
    """
    filepath = os.path.join(path, 'instances.json')
    with open(filepath) as file:
        instances = json.load(file)
        instances = pd.DataFrame(instances['annotations']).set_index('id')

    filename = 'refs(umd).p' if path.endswith('refcocog') else 'refs(unc).p'  # different file name for RefCOCOg
    filepath = os.path.join(path, filename)
    captions = pd.read_pickle(filepath)
    captions = split_sentences(pd.DataFrame(captions))

    captions = pd.merge(captions,
                        instances[['image_id', 'bbox', 'category_id']],
                        left_on='ann_id',
                        right_on='id').set_index('sent_id')

    return captions


def get_refcoco_data(path):
    """fetch data from RefCOCO*

    Args:
        path (string): path to RefCOCO* base dir

    Returns:
        tuple: RefCOCO* data (pd.DataFrame), split IDs (dict -> dict -> list)
    """
    captions = get_refcoco_df(path)

    # partitions: ['train', 'testB', 'testA', 'val']
    partitions = list(pd.unique(captions.refcoco_split))

    image_ids, caption_ids = {}, {}

    for part in partitions:
        image_ids[part] = list(
            captions.loc[captions.refcoco_split == part].image_id.unique())
        caption_ids[part] = captions.loc[captions.refcoco_split ==
                                         part].index.to_list()

    ids = {'image_ids': image_ids, 'caption_ids': caption_ids}

    return (captions, ids)


def split_sentences(df):
    """
        split sentences in refcoco df
    """
    rows = []

    def coco_split(row):
        for split in ['train', 'val', 'test']:
            if split in row['file_name']:
                return split
        return None

    def unstack_sentences(row):
        nonlocal rows
        for i in row.sentences:
            rows.append({
                'sent_id': i['sent_id'],
                'ann_id': row['ann_id'],
                'caption': i['sent'],
                'ref_id': row['ref_id'],
                'refcoco_split': row['split'],
                'coco_split': coco_split(row)
            })

    df.apply(lambda x: unstack_sentences(x), axis=1)

    return pd.DataFrame(rows)


def filename_from_id(image_id, prefix='', file_ending='.jpg'):
    """
    get image filename from id: pad image ids with zeroes,
    add file prefix and file ending
    """
    padded_ids = str(image_id).rjust(12, '0')
    filename = prefix + padded_ids + file_ending

    return (filename)


def crop_image_to_bb(image, bb, return_context=False):
    """
    crop image to bounding box annotated for the current region
    :input:
        Image (PIL Image)
        Bounding Box coordinates (list containing values for x, y, w, h)
    :output:
        Image (PIL Image) cropped to bounding box coordinates
    """

    # convert image into numpy array
    image_array = np.array(image)

    # get bounding box coordinates (round since integers are needed)
    x, y, w, h = round(bb[0]), round(bb[1]), round(bb[2]), round(bb[3])

    # calculate minimum and maximum values for x and y dimension
    x_min, x_max = x, x + w
    y_min, y_max = y, y + h

    # crop image by slicing image array
    target_region = image_array[y_min:y_max, x_min:x_max, :]
    target_mask = np.zeros_like(target_region[:, :, 0]).astype(bool)
    target_image = Image.fromarray(target_region)

    if return_context:
        # mask out target from image and return as context
        # positions with True are not allowed to attend while False values will be unchanged
        context_mask = np.zeros_like(image_array[:, :, 0]).astype(bool)
        image_array[y_min:y_max, x_min:x_max, :] = 0
        context_mask[y_min:y_max, x_min:x_max] = True
        context_image = Image.fromarray(image_array)
        return target_image, target_mask, context_image, context_mask

    return target_image, target_mask


def compute_position_features(image, bb):
    """
    compute position features of bounding box within image
    5 features (all relative to image dimensions):
        - x and y coordinates of bb corner points ((x1,y1) and (x2,y2))
        - bb area
    :input:
        Image (PIL Image)
        Bounding Box coordinates (list containing values for x, y, w, h)
    :output:
        numpy array containing the features computed
    cf. https://github.com/clp-research/clp-vision/blob/master/ExtractFeats/extract.py
    """

    image = np.array(image)
    # get image dimensions, split up list containing bb values
    ih, iw, _ = image.shape
    x, y, w, h = bb

    # x and y coordinates for bb corners
    # upper left
    x1r = x / iw
    y1r = y / ih
    # lower right
    x2r = (x + w) / iw
    y2r = (y + h) / ih

    # bb area
    area = (w * h) / (iw * ih)

    return torch.Tensor([x1r, y1r, x2r, y2r, area])


def pad_img_to_max(image, color=0, centering=(0.5, 0.5)):
    max_dim = max(image.size)
    padded_image = ImageOps.pad(
        image, 
        size=(max_dim, max_dim), 
        centering=centering,
        color=color
    )
    return padded_image


def pad_mask_to_max(mask):

    if type(mask) == np.ndarray:
        mask = torch.from_numpy(mask)
    
    x, y = mask.shape
    if x == y:
        return mask
    min_dim, max_dim = sorted(mask.shape)
    diff = max_dim - min_dim
    pad = diff / 2
    if x > y:  # center on y scale
        return F.pad(mask, (floor(pad),ceil(pad),0,0), 'constant', True)
    else:  # center on x scale
        return F.pad(mask, (0,0,floor(pad),ceil(pad)), 'constant', True)
    

def xywh_to_xyxy(bb):

    x, y, w, h = map(Decimal, map(str, bb))

    # upper left
    x1 = x
    y1 = y
    # lower right
    x2 = x + w
    y2 = y + h

    out = x1, y1, x2, y2

    return tuple(map(float, out))


def xyxy_to_xywh(bb):

    x1, y1, x2, y2 = map(Decimal, map(str, bb))

    w = x2 - x1
    h = y2 - y1

    out = x1, y1, w, h

    return tuple(map(float, out))