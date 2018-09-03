#import argparse
import  os
import torch
from torch import nn
from torchvision import transforms

from main.unet.unet_model import Unet
from main.data.data_loader_implemented import get_data_loader

def train(agrs=''):

    cuda_usage = True

    root_data_path = '../../datasets/cityscapes'

    transform = transforms.Compose([transforms.RandomRotation(10),
                                          transforms.RandomHorizontalFlip()])
    val_loader, train_loader =  get_data_loader(root_data_path, transform, [256, 256], batch_size=32, worker_num=8)

    device = 'cpu'
    if torch.cuda.is_available() and cuda_usage:
        device = 'cuda:1'

    model = Unet()
    model.to(device)
    #model = torch.nn.DataParallel(model, device_ids=[0])
    #model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e04, weight_decay=5e-4)

    criterion = 234 # need to write loss

    for epoch in range(0, 100):
        model.train()
        for i, (images, labels) in enumerate(train_loader):

            # cast data examples to cuda or cpu device
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            #loss = citerion(output, target)# write normal loss

            optimizer.zero_grad()
            #loss.backward()
            optimizer.step()

            #here can be logging

        model.eval()
        for i, (images, labels) in enumerate(val_loader):
            with model.no_grad():
                pass





if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='unet hyperparameters')
    train()