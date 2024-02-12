import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, ToTensor, Resize, ToTensor, ToPILImage, Normalize, Compose
import torchvision as tv
import pandas as pd
import h5py

from PIL import Image, ImageDraw
import numpy as np
import os

from transformers import BertTokenizer

from .utils import crop_image_to_bb, get_paco_df, compute_position_features, pad_img_to_max, pad_mask_to_max, xywh_to_xyxy


class CoverWithNoise:
    
    def __init__(self, noise_coverage=0.5):
        self.noise_coverage = noise_coverage
        
    def __str__(self):
        return f"CoverWithNoise(noise_coverage={self.noise_coverage})"
        
    def __call__(self, image):
        image_tensor = ToTensor()(image)
        # create 1D selection mask
        _, h, w = image_tensor.shape
        mask = np.zeros((h, w), dtype=bool).flatten()  # [hw]

        # sample from flattened index & set mask to True
        idx = np.indices(mask.shape).flatten()
        idx_sample = np.random.choice(idx, replace=False, size=round(mask.size * self.noise_coverage))
        mask[idx_sample] = True
        # reshape to 2D image shape
        mask = torch.from_numpy(mask.reshape((h, w)))  # [h, w]

        # mask image with noise
        noise_tensor = torch.rand_like(image_tensor)
        image_tensor[:, mask] = noise_tensor[:, mask]

        return ToPILImage()(image_tensor)


def get_transforms(mode, config, noise_coverage):
    
    # get default transformations for pretrained model
    model_weights = getattr(tv.models, config.backbone + '_Weights')
    default_transforms = model_weights.DEFAULT.transforms()

    # build train or val transformations
    # (using model defaults for size and normalization)

    resize = Resize(
        size=default_transforms.crop_size,
        interpolation=default_transforms.interpolation
    )
    
    transformations = [
        # add ColorJitter for train mode
        ColorJitter(brightness=[0.5, 1.3],
                    contrast=[0.8, 1.5],
                    saturation=[0.2, 1.5])
        ] if mode == 'train' else []
    
    if noise_coverage > 0:
        transformations += [
            CoverWithNoise(noise_coverage=noise_coverage)
        ]
    
    transformations += [
        # add remaining transformations
        ToTensor(),
        Normalize(mean=default_transforms.mean,
                  std=default_transforms.std)
    ]
    
    transform = Compose(transformations)
    
    return {'resize': resize, 'transform': transform}


def auto_transform(mode, config, noise_coverage):
    if mode.lower() in ['training', 'train']: 
        return get_transforms('train', config, noise_coverage)
    else:
        return get_transforms('val', config, noise_coverage)


class PACODataset(Dataset):

    def __init__(self,
                 data,
                 img_root,
                 max_length,
                 target_transform=None,
                 context_transform=None,
                 return_global_context=False,
                 return_location_features=False,
                 return_scene_features=False,
                 scene_summary_ids=None,
                 scene_summary_features=None,
                 return_tensor=True,
                 ):
        super().__init__()

        self.img_root = img_root
        self.target_transform = target_transform
        self.context_transform = context_transform if context_transform is not None else target_transform
        self.annot = [(entry['id'], entry['file_name'],
                       entry['name'], entry['bbox']) for entry in data]

        assert return_scene_features == False

        # flags for input composition
        self.return_global_context = return_global_context
        self.return_location_features = return_location_features
        self.return_tensor = return_tensor
        self.return_scene_features = return_scene_features
        self.scene_summary_ids = scene_summary_ids
        self.scene_summary_features = scene_summary_features

        # no return_unique in PACODataset
        self.annot_select = self.annot

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.annot_select)
    
    def get_imgs_from_ann_id(self, ann_id):
        annot_dict = dict([(a[0], a[1:]) for a in self.annot_select])
        image_file, caption, bb = annot_dict[ann_id]

        image_filepath = os.path.join(self.img_root, image_file)
        assert os.path.isfile(image_filepath)
        image = Image.open(image_filepath)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        target_image, _, context_image, _ = crop_image_to_bb( # type: ignore
            image, bb, return_context=True)
        
        return image, target_image, context_image, caption
    
    def get_bbox_from_ann_id(self, ann_id):
        annot_dict = dict([(a[0], a[1:]) for a in self.annot_select])
        _, _, bb = annot_dict[ann_id]
        return bb
    
    def get_annotated_image(self, ann_id, return_caption=False, bbox_color='blue', width=3):
        full_image, _, _, caption = self.get_imgs_from_ann_id(ann_id)
        bbox = self.get_bbox_from_ann_id(ann_id)
        bbox_xyxy = xywh_to_xyxy(bbox)
        
        draw = ImageDraw.Draw(full_image)
        draw.rectangle(bbox_xyxy, outline=bbox_color, width=width)
        
        return full_image if not return_caption else (full_image, caption)

    def __getitem__(self, idx):
        ann_id, image_file, caption, bb = self.annot_select[idx]
        image_filepath = os.path.join(self.img_root, image_file)
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

        return ann_id, *encoder_input, caption, cap_mask


def process_part_names(df, only='part'):
    
    parts_df = df.loc[df.supercategory == 'PART']
    
    if only == 'part':
        parts_df['name'] = parts_df['name'].map(lambda x: x.split(':')[-1]) # keep part after ':'
    elif only == 'full':
        parts_df['name'] = parts_df['name'].map(lambda x: x.split(':')[0])  # keep part before ':'
    else:
        raise NotImplementedError(f"'only' value has to by 'part' or 'full', but is {only}")
    
    df.loc[parts_df.index] = parts_df
    
    return df


def build_dataset(config,
                  mode='training',
                  return_unique=False, 
                  return_tensor=True,
                  noise_coverage=0,
                  parts_only_part=False,
                  parts_only_full=False,
                  ):

    assert mode in ['training', 'train', 'validation', 'val', 'test'], f"{mode} not supported"
    if mode == 'training':
        mode = 'train'
    elif mode == 'validation': 
        mode = 'val'
        
    if config.verbose:
        print(f'using data from {config.prefix} {mode}')
        
    # get paco data
    if config.use_normalized_paco:
        paco_ann_file = f'paco_ego4d_v1_{mode}_cleaned.json'
        paco_ann_path = os.path.join(config.paco_base, paco_ann_file)
        data = pd.read_json(paco_ann_path)
    else:
        paco_ann_file = f'paco_ego4d_v1_{mode}.json'
        paco_ann_path = os.path.join(config.paco_base, paco_ann_file)
        data = get_paco_df(paco_ann_path).reset_index()
        
    if parts_only_part:
        data = process_part_names(data, only='part')
    elif parts_only_full:
        data = process_part_names(data, only='full')
    
    # set target and context transformation according to mode
    target_transform = auto_transform(mode, config, noise_coverage)
    context_transform = auto_transform(mode, config, 0)  # no noise for context
    
    # handle scene summaries if set in config
    if vars(config).get('use_scene_summaries', False):
        raise NotImplementedError('Scene Summaries are not implemented')
        # scenesum_filepath = os.path.join(
        #     config.project_data_path, 'scene_summaries', f'scene_summaries_{partition}.h5')
        # print(f'read scene summaries from {scenesum_filepath}')
        # with h5py.File(scenesum_filepath,'r') as f:
        #     scenesum_ann_ids = f['ann_ids'][:].squeeze(1)
        #     scenesum_feats = f['context_feats'][:]
    else: 
        scenesum_ann_ids = scenesum_feats = None
        
    # build dataset
    dataset = PACODataset(data=data.to_dict(orient='records'),
                             img_root=config.paco_imgs,
                             max_length=config.max_position_embeddings,
                             target_transform=target_transform,
                             context_transform=context_transform,
                             return_global_context=vars(config).get('use_global_features', False),
                             return_location_features=vars(config).get('use_location_features', False), 
                             return_scene_features=vars(config).get('use_scene_summaries', False),
                             scene_summary_ids=scenesum_ann_ids,
                             scene_summary_features=scenesum_feats,

                             return_tensor=return_tensor)
    
    if config.verbose:
        print(
            f'Initialize {dataset.__class__.__name__} with mode: {mode}', 
            '\ntarget transformation:', target_transform, 
            '\ncontext transformation:', context_transform,
            f'\nentries: {len(dataset)}',
            '\nreturn unique (without function in PACO):', return_unique,
            '\nreturn only part names (for parts):', parts_only_part,
            '\nreturn only full names (for parts):', parts_only_full, 
            f'\nuse normalized PACO: {config.use_normalized_paco}',
            '\n'
        )
    
    return dataset
