from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import Lambda, ColorJitter, RandomHorizontalFlip, ToTensor, Resize, CenterCrop, ToTensor, Normalize, Compose 
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, crop_image_to_bb, get_refcoco_data, compute_position_features

MAX_DIM = 224 # 299


def under_max(image):

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:

    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


def get_transforms(mode, config):
    # get default transformations for pretrained model
    model_weights = getattr(tv.models, config.backbone + '_Weights')
    default_transforms = model_weights.DEFAULT.transforms()

    # build train or val transformations
    # (using model defaults for size and normalization)
    if mode == 'train': 
        transform = Compose([
            RandomRotation(),  # NOTE good idea for REG?
            Lambda(under_max),
            ColorJitter(brightness=[0.5, 1.3],
                        contrast=[0.8, 1.5],
                        saturation=[0.2, 1.5]),
            RandomHorizontalFlip(),  # NOTE good idea for REG?
            Resize(size=default_transforms.resize_size, 
                   interpolation=default_transforms.interpolation),
            CenterCrop(size=default_transforms.crop_size),
            ToTensor(),
            Normalize(mean=default_transforms.mean, 
                      std=default_transforms.std),
        ])
    elif mode == 'val':
        transform = Compose([
            Lambda(under_max),
            Resize(size=default_transforms.resize_size, 
                   interpolation=default_transforms.interpolation),
            CenterCrop(size=default_transforms.crop_size),
            ToTensor(),
            Normalize(mean=default_transforms.mean, 
                      std=default_transforms.std),
        ])
    else:
        raise NotImplementedError(f'transforms mode {mode} is not implemented')
    
    return transform


def auto_transform(mode, config):
    if mode.lower() in ['training', 'train']: 
        return get_transforms('train', config)
    else:
        return get_transforms('val', config)


class RefCocoCaption(Dataset):

    def __init__(self,
                 data,
                 root,
                 max_length,
                 target_transform=None,
                 context_transform=None,
                 return_unique=False,
                 return_global_context=False,
                 return_location_features=False,
                 return_nested_tensor=True
                 ):
        super().__init__()

        self.root = root
        self.target_transform = target_transform
        self.context_transform = context_transform if context_transform is not None else target_transform
        self.annot = [(entry['ann_id'], self._process(entry['image_id']),
                       entry['caption'], entry['bbox']) for entry in data]
        self.return_global_context = return_global_context
        self.return_location_features = return_location_features
        self.return_nested_tensor = return_nested_tensor

        if return_unique:
            # filter for unique ids
            self.annot_select = []
            stored_ids = []
            for a in self.annot:
                if a[0] not in stored_ids:
                    self.annot_select.append(a)
                    stored_ids.append(a[0])
        else:
            self.annot_select = self.annot

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower=True)
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return 'COCO_train2014_' + val + '.jpg'

    def __len__(self):
        return len(self.annot_select)

    def __getitem__(self, idx):
        ann_id, image_file, caption, bb = self.annot_select[idx]
        image_filepath = os.path.join(self.root, 'train2014', image_file)
        assert os.path.isfile(image_filepath)

        image = Image.open(image_filepath)

        # CAPTION

        caption_encoded = self.tokenizer.encode_plus(
            caption,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 -
                    np.array(caption_encoded['attention_mask'])).astype(bool)

        # IMAGE

        # convert if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # with transforms: proceed with building encoder input
        #   if global=False, location=False:
        #       encoder_input = t_img, t_mask
        #   if global=True, location=False:
        #       encoder_input = t_img, t_mask, g_img, g_mask
        #   if global=False, location=True:
        #       encoder_input = t_img, t_mask, loc
        #   if global=True, location=True:
        #       encoder_input = t_img, t_mask, g_img, g_mask, loc

        # target bb
        target_image, context_image = crop_image_to_bb(
            image, bb, return_context=True)
        target_image = self.target_transform(target_image)
        if self.return_nested_tensor:
            target_image = nested_tensor_from_tensor_list(
                target_image.unsqueeze(0))
            encoder_input = [
                target_image.tensors.squeeze(0),
                target_image.mask.squeeze(0)
                ]
        else:
            # for returning non-tensor images / visualization
            encoder_input = [target_image]

        if self.return_global_context:
            # add global context
            context_image = self.context_transform(context_image)
            if self.return_nested_tensor:
                context_image = nested_tensor_from_tensor_list(
                    context_image.unsqueeze(0))
                encoder_input += [
                    context_image.tensors.squeeze(0),
                    context_image.mask.squeeze(0)
                    ]
            else:
                # for returning non-tensor images / visualization
                encoder_input += [context_image]

        if self.return_location_features:
            # add location features
            position_feature = compute_position_features(image, bb)
            encoder_input.append(position_feature)

        return ann_id, *encoder_input, caption, cap_mask


def build_dataset(config,
                  mode='training',
                  transform='auto',
                  return_unique=False, 
                  return_nested_tensor=True):

    assert mode in ['training', 'train', 'validation', 'val', 'testa', 'testb']

    # get refcoco data
    full_data, ids = get_refcoco_data(config.ref_dir)

    # select data partition
    if mode.lower() in ['training', 'train']:
        data = full_data.loc[ids['caption_ids']['train']]
    elif mode.lower() in ['validation', 'val']:
        data = full_data.loc[ids['caption_ids']['val']]
    elif mode.lower() == 'testa':
        data = full_data.loc[ids['caption_ids']['testA']]
    elif mode.lower() == 'testb':
        data = full_data.loc[ids['caption_ids']['testB']]
    else:
        raise NotImplementedError(f"{mode} not supported")
    
    # parse transform parameter
    if transform == 'auto':
        # set target and context transformation according to mode
        transform = auto_transform(mode, config)
        target_transform, context_transform = transform, transform
    elif type(transform) == Compose:
        # assign transform to both target_transform and context_transform
        target_transform, context_transform = transform, transform
    elif type(transform) == dict:
        # for different transformation settings
        assert set(transform.keys()) == {'context', 'target'}
        for key in transform.keys():
            if transform[key] == 'auto':
                transform[key] = auto_transform(mode, config)
        target_transform = transform['target']
        context_transform = transform['context']
    else:
        raise ValueError
    
    if config.verbose:
        print(f'Initialize Dataset with mode: {mode}', 
            '\ntarget transformation:', target_transform, 
            '\ncontext transformation:', context_transform,
            f'\nentries: {len(data)}',
            '\nreturn unique:', return_unique, '\n')

    # build dataset
    dataset = RefCocoCaption(data=data.to_dict(orient='records'),
                             root=config.dir,
                             max_length=config.max_position_embeddings,
                             target_transform=target_transform,
                             context_transform=context_transform,
                             return_unique=return_unique,
                             return_global_context=config.use_global_features,
                             return_location_features=config.use_location_features, 
                             return_nested_tensor=return_nested_tensor)
    return dataset
