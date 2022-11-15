from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, crop_image_to_bb, get_refcoco_data

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

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
                 return_unique=False):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(entry['ann_id'], self._process(entry['image_id']),
                       entry['caption'], entry['bbox']) for entry in data]

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

        # crop image to bounding box
        image = crop_image_to_bb(image, bb)

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 -
                    np.array(caption_encoded['attention_mask'])).astype(bool)

        return ann_id, image.tensors.squeeze(0), image.mask.squeeze(
            0), caption, cap_mask


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
                             return_unique=return_unique)
    return dataset
