#import argparse
import torch
from torchvision import transforms
from torch.utils import data
from main.unet.unet_model import Unet
from main.data.data_loader_implemented import cityscapesLoader

def train(agrs=''):
    root_data_path = '../../datasets/cityscapes'

    transforms_train = transforms.Compose([transforms.RandomRotation(10),
                                          transforms.RandomHorizontalFlip()])

    dataloader_trn = cityscapesLoader(root_data_path, img_size=(256, 256), is_transform=True, augmentations=transforms_train)
    dataloader_val = cityscapesLoader(root_data_path, img_size=(256, 256), is_transform=True, augmentations=transforms_train, split='val')

    valloader = data.DataLoader(dataloader_val, batch_size=32, num_workers=8)
    trainloader = data.DataLoader(dataloader_trn, batch_size=32, shuffle=True, num_workers=8)

    model = Unet()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e04, weight_decay=5e-4)

    loss = 234 # need to write loss

    for epoch in range(0, 100):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            images = torch.autograd.Variable(images.cuda())
            labels = torch.autograd.Variable(labels.cuda())
            optimizer.zero_grad()

            ouput = model(images)
            loss = 344 # write normal loss

            #loss.backward()
            optimizer.step()
            #print

        model.eval()
        for i, (images, labels) in enumerate(valloader):
            pass





if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='unet hyperparameters')
    train()