import torch
import os

def get_latest_checkpoint(config):
    """get latest checkpoint from defined location"""

    available_cpts = os.listdir(config.checkpoint_path)

    if len(available_cpts) > 0:
        available_epochs = [
            int(os.path.splitext(cpt)[0].split('_')[-1]) for cpt in available_cpts]
        highest_epoch = max(available_epochs)
        return f'{config.prefix}_checkpoint_{highest_epoch}.pth'
    else:
        return None

def save_ckp(epoch, model, optimizer, lr_scheduler, train_loss, val_loss, path):
    """save training checkpoint"""

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, path)


def load_ckp(model, optimizer, lr_scheduler, path):
    """load training checkpoint"""

    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    return epoch, model, optimizer, lr_scheduler, train_loss, val_loss