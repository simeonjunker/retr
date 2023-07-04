import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import Lambda, ColorJitter, RandomHorizontalFlip, ToTensor, Resize, CenterCrop, ToTensor, Normalize, Compose
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os
import h5py

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, crop_image_to_bb, get_refcoco_data, compute_position_features, pad_img_to_max, pad_mask_to_max


def get_transforms(mode, config):
    # get default transformations for pretrained model
    model_weights = getattr(tv.models, config.backbone + '_Weights')
    default_transforms = model_weights.DEFAULT.transforms()

    # build train or val transformations
    # (using model defaults for size and normalization)

    resize = Resize(
        size=default_transforms.crop_size,
        interpolation=default_transforms.interpolation
    )

    if mode == 'train': 
        transform = Compose([
            ColorJitter(brightness=[0.5, 1.3],
                        contrast=[0.8, 1.5],
                        saturation=[0.2, 1.5]),
            ToTensor(),
            Normalize(mean=default_transforms.mean, 
                      std=default_transforms.std),
        ])
        
    elif mode == 'val':
        transform = Compose([
            ToTensor(),
            Normalize(mean=default_transforms.mean, 
                      std=default_transforms.std),
        ])
    else:
        raise NotImplementedError(f'transforms mode {mode} is not implemented')
    
    return {'resize': resize, 'transform': transform}


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
                 return_tensor=True,
                 return_scene_features=False,
                 scene_summary_ids=None,
                 scene_summary_features=None
                 ):
        super().__init__()

        self.root = root
        self.target_transform = target_transform
        self.context_transform = context_transform if context_transform is not None else target_transform
        self.annot = [(entry['ann_id'], self._process(entry['image_id']),
                       entry['caption'], entry['bbox']) for entry in data]

        # TODO remove this after testing!
        self.annot = [a for a in self.annot if a[0] not in [599584, 1617100, 1903307, 1963185, 2191866, 2227390, 1616547, 1619639]]


        self.return_global_context = return_global_context
        self.return_location_features = return_location_features
        self.return_tensor = return_tensor
        self.return_scene_features = return_scene_features
        self.scene_summary_ids = scene_summary_ids
        self.scene_summary_features = scene_summary_features

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

        # crop to bounding box
        target_image, target_mask, context_image, context_mask = crop_image_to_bb(
            image, bb, return_context=True)

        # with transforms: proceed with building encoder input
        #   if global=False, location=False, scene=False:
        #       encoder_input = t_img, t_mask
        #   if global=True, location=False, scene=False:
        #       encoder_input = t_img, t_mask, g_img, g_mask
        #   if global=False, location=True, scene=False:
        #       encoder_input = t_img, t_mask, loc
        #   if global=True, location=True, scene=False:
        #       encoder_input = t_img, t_mask, g_img, g_mask, loc
        #   if global=False, location=True, scene=True:
        #       encoder_input = t_img, t_mask, loc, scene

        # target bb
        target_image = pad_img_to_max(target_image)
        target_image = self.target_transform['resize'](target_image)
        target_image = self.target_transform['transform'](target_image)
        
        target_mask = pad_mask_to_max(target_mask)
        target_mask = self.target_transform['resize'](target_mask.unsqueeze(0))

        if self.return_tensor:
            encoder_input = [
                target_image,
                target_mask.squeeze(0)
                ]

        else:
            # for returning non-tensor images / visualization
            encoder_input = [target_image]

        if self.return_global_context:
            # add global context
            context_image = pad_img_to_max(context_image)
            context_image = self.context_transform['resize'](context_image)            
            context_image = self.context_transform['transform'](context_image)
            
            context_mask = pad_mask_to_max(context_mask)
            context_mask = self.context_transform['resize'](context_mask.unsqueeze(0))

            if self.return_tensor:
                encoder_input += [
                    context_image,
                    context_mask.squeeze(0)
                    ]

            else:
                # for returning non-tensor images / visualization
                encoder_input += [context_image]

        if self.return_location_features:
            # add location features
            position_feature = compute_position_features(image, bb)
            encoder_input.append(position_feature)

        if self.return_scene_features: 
            # add scene summaries
            selection_mask = self.scene_summary_ids==ann_id
            scene_summary = torch.from_numpy(
                self.scene_summary_features[selection_mask]).squeeze()
            encoder_input.append(scene_summary)

        return ann_id, *encoder_input, caption, cap_mask


def build_dataset(config,
                  mode='training',
                  transform='auto',
                  return_unique=False, 
                  return_tensor=True):

    assert mode in ['training', 'train', 'validation', 'val', 'testa', 'testb', 'test']

    # get refcoco data
    if config.verbose:
        print(f'using data from {config.prefix}')

    full_data, ids = get_refcoco_data(config.ref_dir)

    # select data partition
    
    if mode.lower() in ['training', 'train']:
        partition = 'train'
    elif mode.lower() in ['validation', 'val']:
        partition = 'val'
    elif mode.lower() == 'testa':  # refcoco / refcoco+
        partition = 'testA'
    elif mode.lower() == 'testb':  # refcoco / refcoco+
        partition = 'testB'
    elif mode.lower() == 'test':  # refcocog
        partition = 'test'
    else:
        raise NotImplementedError(f"{mode} not supported")
    
    data = full_data.loc[ids['caption_ids'][partition]]
    
    # parse transform parameter
    if transform == 'auto':
        # set target and context transformation according to mode
        transform = auto_transform(mode, config)
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
        raise NotImplementedError('input for transform parameter has to be "auto" or dict with transforms for "context" and "target"')
    
    if config.verbose:
        print(f'Initialize Dataset with mode: {partition}', 
            '\ntarget transformation:', target_transform, 
            '\ncontext transformation:', context_transform,
            f'\nentries: {len(data)}',
            '\nreturn unique:', return_unique, '\n')

    # handle scene summaries if set in config
    if config.use_scene_summaries:
        scenesum_filepath = os.path.join(
            config.project_data_path, 'scene_summaries', f'scene_summaries_{partition}.h5')
        with h5py.File(scenesum_filepath,'r') as f:
            scenesum_ann_ids = f['ann_ids'][:].squeeze(1)
            scenesum_feats = f['context_feats'][:]
    else: 
        scenesum_ann_ids = scenesum_feats = None

    # build dataset
    dataset = RefCocoCaption(data=data.to_dict(orient='records'),
                             root=config.dir,
                             max_length=config.max_position_embeddings,
                             target_transform=target_transform,
                             context_transform=context_transform,
                             return_unique=return_unique,
                             return_global_context=config.use_global_features,
                             return_location_features=config.use_location_features, 
                             return_tensor=return_tensor, 
                             return_scene_features=config.use_scene_summaries,
                             scene_summary_ids=scenesum_ann_ids,
                             scene_summary_features=scenesum_feats)
    return dataset
