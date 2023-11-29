import torch
import os

def save_ckp(epoch, model, optimizer, lr_scheduler, args, config, train_loss, val_loss, cider_score, path):
    """save training checkpoint"""

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'args': args,
        'config': config,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'cider_score': cider_score
    }, path)


def load_ckp(model, optimizer, lr_scheduler, path):
    """load training checkpoint"""

    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    epoch = checkpoint['epoch']
    args = checkpoint['args']
    config = checkpoint['config']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    cider_score = checkpoint['cider_score']

    return epoch, model, optimizer, lr_scheduler, args, config, train_loss, val_loss, cider_score