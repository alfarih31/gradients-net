import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import model

TRAIN_FOLDER = '/home/alfarihfz/data/Pulmonary/train/'
BATCH_SIZE=64
EPOCH = 500
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image_loader(path):
    return Image.open(path)

transform = transforms.Compose(
    [
    transforms.ToTensor(),
    ]
)

gnet = model.GNet(device=DEVICE)
hog = model.HOGLayer(device=DEVICE)

trainData = datasets.DatasetFolder(root=TRAIN_FOLDER,
                                    loader=image_loader,
                                    extensions=['jpeg', 'png'],
                                    transform=transform)
dataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.Adam(gnet.parameters(), weight_decay=5e-4)
criterion = torch.nn.MSELoss()

for epoch in range(EPOCH):
    batch_loss = 0.0
    for i, data in enumerate(dataLoader):
        img,_ = data
        img.to(DEVICE)

        optimizer.zero_grad()

        predicts = gnet(img)
        labels = hog(img)

        loss = criterion(predicts, labels)

        loss.backward()
        optimizer.step()

        batch_loss += loss.item()

        if i%16 == 0:
            print('Epoch %d | Batch %d Loss: %.4f'%(epoch, i, batch_loss))

    if epoch % 10 == 0 and epoch != 0:
        torch.save(gnet.state_dict(), 'GNET_%d.pth'%epoch)
        print('SAVING MODEL GNET_%d.pth'%epoch)

