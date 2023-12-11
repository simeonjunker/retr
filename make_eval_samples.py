from pathlib import Path
import argparse
import torch
from tqdm import tqdm
import os
import os.path as osp
import json
import csv
from random import sample, seed
from copy import deepcopy
import base64
from io import BytesIO
from glob import glob
from torch.utils.data import DataLoader, SequentialSampler

from models import caption
from configuration import Config
from data_utils import refcoco
from engine import pack_encoder_inputs
from eval_utils.decode import greedy_decoding
from eval_model import noise_from_checkpoint
from inference.inference_utils import prepare_tokenizer, setup_val_dataloader, override_config_with_checkpoint

PROJECT_PATH = osp.dirname(osp.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENTRY_TEMPLATE = """

<p>

<div>
<img src="data:image/jpeg;base64, {b64_image}"/>

<br>

<table>
  <tr>
    <td>index:</td>
    <td>{index}</td>
  </tr>
  <tr>
    <td>ann_id:<td>
    <td>{ann_id}</td>
  </tr>
  <tr>
    <td>generated:</td>
    <td>{generated}</td>
  </tr>
  <tr>
    <td>annotated:</td>
    <td>{annotated}</td>
  </tr>
</table> 
</div>

</p>

<hr>
"""

HTML_TEMPLATE = """ <!DOCTYPE html>
    <html>
    <head>
    <title>Page Title</title>
    </head>
    <body>

    {body}

    </body>
    </html> """

def img_to_b64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    b64_str = base64.b64encode(buffered.getvalue()).decode('ASCII')
    return b64_str


def main(args, local_config):
    
    model_paths = sorted(glob(osp.join(args.input_dir, '**/**.pth')))
    print('found following models: ', model_paths)
    
    if not os.path.isdir(args.output_dir):
        print(f'create output directory {args.output_dir}')
        os.makedirs(args.output_dir)
    
    for model_checkpoint in tqdm(model_paths):
        
        # create output paths and check whether files already exist
        checkpoint_name = osp.split(model_checkpoint)[-1]
        print(f'checkpoint name: {checkpoint_name}')
        html_outfile = osp.join(args.output_dir, checkpoint_name.replace('.pth', f'_{args.split}_sample.html'))
        csv_outfile = osp.join(args.output_dir, checkpoint_name.replace('.pth', f'_{args.split}_sample_anns.csv'))
        
        if os.path.isfile(html_outfile) and os.path.isfile(csv_outfile) and not args.overwrite_existing_files:
            print(f'files {html_outfile} and {csv_outfile} already exist -- CANCEL')
            break
        
        # build config
        checkpoint_data = torch.load(model_checkpoint, map_location="cpu")
        
        if 'config' in checkpoint_data.keys():
            print('using config from checkpoint')
            config = checkpoint_data['config']
            config.dir = local_config.dir
            config.ref_base = local_config.ref_base
            config.ref_dir = local_config.ref_dir
        else:
            print('using local config')
            config = local_config
            if args.override_config:
                override_config_with_checkpoint(osp.split(args.checkpoint)[-1], config)

        config.batch_size = 1  # override batch size
        
        # build model        
        if not os.path.exists(model_checkpoint):
            raise NotImplementedError("Give valid checkpoint path")
        else:
            model, _ = caption.build_model(config)
            checkpoint = torch.load(model_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(DEVICE)
            model.eval()
        
        noise = noise_from_checkpoint(checkpoint_name)
        assert noise is not None

        print(f'Successfully loaded {model.__class__.__name__} model with noise level {noise}')
        
        tokenizer, _, _ = prepare_tokenizer()
        pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        bos_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        
        data_loader = setup_val_dataloader(config, noise_coverage=noise, split=args.split)

        # dataset
        dataset = refcoco.build_dataset(
            config, 
            mode=args.split, 
            noise_coverage=noise,
            return_unique=True)
        
        # copy for sampling
        sample_dataset = deepcopy(dataset)

        sample_outfile = osp.join(args.output_dir, f'{config.prefix}-{args.split}-sample-{args.k}.json')
        if osp.isfile(sample_outfile):
            with open(sample_outfile, 'r') as f:
                ann_ids_sample = json.load(f)
                print('read sample from file')
        else:
            print('make new sample')
            ann_ids = sorted([a[0] for a in dataset.annot_select])

            # sample k entries from dataset
            seed(args.random_seed)
            ann_ids_sample = sorted(sample(ann_ids, args.k))
            
            with open(sample_outfile, 'w') as f:
                json.dump(ann_ids_sample, f)
                print('write sample to file')

        # restrict dataset to sampled entries
        annot_select_sample = [a for a in dataset.annot_select if a[0] in ann_ids_sample]
        sample_dataset.annot_select = sorted(annot_select_sample, key=lambda x: x[0])
        print(f'sampled {len(sample_dataset)} from dataset with {len(dataset)} entries')

        # build dataloader
        sampler_val = SequentialSampler(sample_dataset)
        data_loader = DataLoader(
            sample_dataset,
            batch_size=config.batch_size,
            sampler=sampler_val,
            drop_last=False,
            num_workers=config.num_workers,
        )
        global_features = data_loader.dataset.return_global_context
        location_features = data_loader.dataset.return_location_features

        results = []

        # generate samples with model
        for i, (ann_ids, *encoder_input, _, _) in tqdm(enumerate(data_loader)): 

            samples = pack_encoder_inputs(
                encoder_input, global_features, location_features)

            # get model predictions
            generated = greedy_decoding(
                samples, model, tokenizer,
                max_len=config.max_position_embeddings, clean=True,
                pad_token=pad_id, bos_token=bos_id, eos_token=eos_id,
                device='auto'
            )
            
            assert len(generated) == 1
            out = {"ann_id": ann_ids.item(), "generated": generated[0]}
            if args.print_samples:
                print(out)
            
            results.append(out)
            
        # generate html file with samples
        html_entries = []

        for i, sample_dict in enumerate(results):
            ann_id = sample_dict['ann_id']
            generated = sample_dict['generated']
            image, annotated = sample_dataset.get_annotated_image(ann_id, return_caption=True)
            b64_image = img_to_b64(image)
            
            html_entry = ENTRY_TEMPLATE.format(
            b64_image=b64_image,
            index=i,
            ann_id=ann_id,
            generated=generated,
            annotated=annotated
            ).strip()
            
            html_entries.append(html_entry)
            
        entries_str = '\n'.join(html_entries)

        with open(html_outfile, 'w') as f:
            f.write(HTML_TEMPLATE.format(body=entries_str))
            
        # generate csv file for annotation
        with open(csv_outfile, 'w', newline='') as csvfile:
            
            fieldnames = ['idx', 'ann_id', 'generated', 'annotation']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, sample_dict in enumerate(results):
                ann_id = sample_dict['ann_id']
                generated = sample_dict['generated']
            
                writer.writerow({'idx': i, 'ann_id': ann_id, 'generated': generated, 'annotation': ''})
                
        print(f'{checkpoint_name}: DONE')
            

if __name__ == '__main__':
    
    local_config = Config()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', default=osp.join(PROJECT_PATH, 'checkpoints'))
    parser.add_argument('--split', default='val')
    parser.add_argument('--output_dir', default=osp.join(PROJECT_PATH, 'generated', 'identification_samples'))
    parser.add_argument('--overwrite_existing_files', action='store_true')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--k', default=100, type=int)
    parser.add_argument('--print_samples', action='store_true')
    
    args = parser.parse_args()
    
    
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    
    main(args, local_config)