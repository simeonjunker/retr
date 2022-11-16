import torch
from glob import glob
from os.path import dirname, abspath
import matplotlib.pyplot as plt

file_path = dirname(abspath(__file__))
files = sorted(glob(f'{file_path}/coco/**.pth'))

epochs = []
train_loss = []
val_loss = []
ciders = []

for file in files:
    try:
        data = torch.load(file, map_location=torch.device('cpu'))
        
        epochs.append(data['epoch'])
        train_loss.append(data['train_loss'])
        val_loss.append(data['val_loss'])
        ciders.append(data['cider_score'])
    except:
        print(f'invalid file: {file}')

plt.Figure()
plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val_loss')
plt.plot(ciders, label='cider scores')
plt.xticks(epochs)
plt.legend()
plt.grid()
plt.show()