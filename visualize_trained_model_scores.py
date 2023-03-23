import os
import torch
import argparse
import matplotlib.pyplot as plt
from glob import glob


def scores_from_checkpoint(path):

    checkpoint = torch.load(path, map_location='cpu')
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    cider_score = checkpoint['cider_score']

    return epoch, train_loss, val_loss, cider_score


def process_cider(epochs, scores):

    fig = plt.figure()
    plt.plot(epochs, scores, marker='o')
    score_range = max(scores) - min(scores)
    ymin = min(scores) - score_range / 10
    ymax = max(scores) + score_range / 10

    for idx, (epoch, score) in enumerate(zip(epochs, scores)):
        if score > max([0]+scores[:idx]):
            plt.vlines(x = epoch, ymin = ymin, ymax = score, linestyles='dashed', colors='red')

    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('CIDEr')
    plt.xticks(epochs)
    plt.ylim(ymin, ymax)

    best_idx = scores.index(max(scores))
    best = epochs[best_idx]

    txt=f'Best epoch: {best}'
    plt.figtext(0.5, -0.05, txt, wrap=True, horizontalalignment='center')

    return fig


def process_loss(epochs, scores):

    fig = plt.figure()
    plt.plot(epochs, scores, marker='o')
    score_range = max(scores) - min(scores)
    ymin = min(scores) - score_range / 10
    ymax = max(scores) + score_range / 10


    for idx, (epoch, score) in enumerate(zip(epochs, scores)):
        if score < min([1000]+scores[:idx]):
            plt.vlines(x = epoch, ymin = ymin, ymax = score, linestyles='dashed', colors='red')

    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('CIDEr')
    plt.xticks(epochs)
    plt.ylim(ymin, ymax)

    best_idx = scores.index(min(scores))
    best = epochs[best_idx]

    txt=f'Best epoch: {best}'
    plt.figtext(0.5, -0.05, txt, wrap=True, horizontalalignment='center')

    return fig


def main(partial_name):
    partial_name = os.path.abspath(partial_name)

    print('Read data from checkpoints...')
    ckpt_files = sorted(glob(partial_name + '**'))
    ckpt_data = []
    for f in ckpt_files:
        epoch, train_loss, val_loss, cider_score = scores_from_checkpoint(f)
        ckpt_data.append({
            'epoch' : epoch,
            'train_loss' : train_loss,
            'val_loss' : val_loss,
            'cider_score' : cider_score
        })

    epochs = [c['epoch'] for c in ckpt_data]
    train_losses = [c['train_loss'] for c in ckpt_data]
    val_losses = [c['val_loss'] for c in ckpt_data]
    cider_scores = [c['cider_score'] for c in ckpt_data]
    
    print('Compute figures...')
    cider_figure = process_cider(epochs, cider_scores)
    train_loss_figure = process_loss(epochs, train_losses)
    val_loss_figure = process_loss(epochs, val_losses)

    print(f'Save figures to {os.path.split(partial_name)[0]}...')
    cider_figure.savefig(os.path.join(partial_name + 'cider_scores_per_epoch.jpg'), bbox_inches='tight')
    train_loss_figure.savefig(os.path.join(partial_name + 'train_loss_scores_per_epoch.jpg'), bbox_inches='tight')
    val_loss_figure.savefig(os.path.join(partial_name + 'val_loss_scores_per_epoch.jpg'), bbox_inches='tight')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('partial_name')
    args = parser.parse_args()

    main(args.partial_name)