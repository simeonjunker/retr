from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, crop_image_to_bb, get_refcoco_data, compute_position_features

MAX_DIM = 299


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


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3],
                              contrast=[0.8, 1.5],
                              saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class RefCocoCaption(Dataset):

    def __init__(self,
                 data,
                 root,
                 max_length,
                 limit,
                 transform=train_transform,
                 return_unique=False,
                 return_global_context=False,
                 return_location_features=False
                 ):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(entry['ann_id'], self._process(entry['image_id']),
                       entry['caption'], entry['bbox']) for entry in data]
        self.return_global_context = return_global_context
        self.return_location_features = return_location_features

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
        image = Image.open(os.path.join(self.root, 'train2014', image_file))

        # convert if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # target bb
        target_image = crop_image_to_bb(image, bb)
        if self.transform:
            target_image = self.transform(target_image)
        target_image = nested_tensor_from_tensor_list(target_image.unsqueeze(0))
        encoder_input = [target_image.tensors.squeeze(0), target_image.mask.squeeze(0)]

        # global context
        if self.return_global_context:
            global_image = image
            if self.transform:
                global_image = self.transform(global_image)
            global_image = nested_tensor_from_tensor_list(global_image.unsqueeze(0))
            encoder_input += [global_image.tensors.squeeze(0), global_image.mask.squeeze(0)]

        # location features
        if self.return_location_features:
            position_feature = compute_position_features(image, bb)
            encoder_input.append(position_feature)

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

        # encoder_input: 
        #   if global=False, location=False: t_img, t_mask
        #   if global=True, location=False: t_img, t_mask, g_img, g_mask
        #   if global=False, location=True: t_img, t_mask, loc
        #   if global=True, location=True: t_img, t_mask, g_img, g_mask, loc
        return ann_id, *encoder_input, caption, cap_mask


def build_dataset(config,
                  mode='training',
                  transform='auto',
                  return_unique=False):

    full_data, ids = get_refcoco_data(config.ref_dir)

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

    if transform == 'auto':
        transform = train_transform if mode.lower() in ['training', 'train'
                                                        ] else val_transform

    dataset = RefCocoCaption(data=data.to_dict(orient='records'),
                             root=config.dir,
                             max_length=config.max_position_embeddings,
                             limit=config.limit,
                             transform=transform,
                             return_unique=return_unique,
                             return_global_context=config.use_global_features,
                             return_location_features=config.use_location_features)
    return dataset
