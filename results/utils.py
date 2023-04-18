import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from skimage import io
from os.path import join


def parse_filename(filename):
    condition, epoch, split, _, noise_level = re.findall(r'refcoco_(loc|loc_glob|loc_scene)_checkpoint_(\d+)_(val|testa|testb)_(noise(\d-\d+))?', filename)[0]
    noise_level = 0 if noise_level == '' else float(noise_level.replace('-', '.'))
    return condition, epoch, split, noise_level


def filename_from_id(image_id, prefix='', file_ending='.jpg'):
    """
    get image filename from id: pad image ids with zeroes,
    add file prefix and file ending
    """
    padded_ids = str(image_id).rjust(12, '0')
    filename = prefix + padded_ids + file_ending

    return (filename)

def patches_from_bb(e, linewidth=2, edgecolor="g", facecolor="none"):
    bbox = e.bbox
    # Create a Rectangle patch
    return patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2],
        bbox[3],
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )


def display_target_distractors(
    entry_id,
    refcoco_data,
    coco_base,
    linewidth=2,
    target_color="g",
    distractor_color="r",
    facecolor="none",
    dpi='figure',
    save=False
):
    
    # get entry
    entry = refcoco_data.loc[entry_id]
    
    # Create figure and axes
    _, ax = plt.subplots(figsize=(10, 10))

    # Retrieve & display the image
    image_file = filename_from_id(entry.image_id, prefix='COCO_train2014_')
    image_filepath = join(coco_base, 'train2014', image_file)
    image = io.imread(image_filepath)
    ax.imshow(image)

    # Add the patch to the Axes
    ax.add_patch(
        patches_from_bb(
            entry, edgecolor=target_color, linewidth=linewidth, facecolor=facecolor
        )
    )

    if type(refcoco_data) == pd.core.frame.DataFrame:

        # add patches for distractors
        add_patch = lambda x: ax.add_patch(
            patches_from_bb(
                x, edgecolor=distractor_color, linewidth=linewidth, facecolor=facecolor
            )
        )

        distractors = (
            refcoco_data.loc[refcoco_data.image_id == entry.image_id]
            .loc[refcoco_data.ann_id != entry.ann_id]
            .groupby("ann_id")
            .first()
        )

        distractors.apply(lambda x: add_patch(x), axis=1)
        
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    if save:
        print(entry)
        filename = f'example_{entry.image_id}_{entry.name}.jpg'
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        print(f'saved to {filename}')

    plt.show()
    
    
def display_entry(entry_id, refcoco_data, **kwargs):
    
    display_target_distractors(entry_id, refcoco_data, **kwargs)
    
    entry = refcoco_data.loc[entry_id]
    entry_sents = [s['sent'] for s in entry.sentences]
    
    for sent in entry_sents:
        print(sent)
        
    print('\n')
    print('ann_id:\t', entry.ann_id)
    print('img_id:\t', entry.image_id)
    print('license:', entry.license)
    print('flickr:\t', entry.flickr_url)