#import argparse
import torch
#import numpy as np
from torch import nn
from torchvision import transforms



from main.src.unet.unet_model import Unet
from main.data.data_loader_implemented import get_data_loader
from main.src.unet.loss import IoU


def train(agrs=''):
    batch_szie = 1
    img_size = [256, 256]
    worker_num = 8
    cuda_usage = True

    root_data_path = '../../../../../datasets/cityscapes'

    transform = transforms.Compose([transforms.RandomRotation(10),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize(img_size),
                                    transforms.ToTensor()])



    val_loader, train_loader = get_data_loader(root_data_path, transform, img_size, batch_size=batch_szie,
                                               worker_num=worker_num)

    device = 'cpu'
    if torch.cuda.is_available() and cuda_usage:
        device = 'cuda:1'

    model = Unet()
    model.to(device)

    #model = torch.nn.DataParallel(model, device_ids=[0])
    #model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e04, weight_decay=5e-4)

    criterion = IoU().to(device) #nn.CrossEntropyLoss()# need to write criterion

    for epoch in range(0, 100):
        model.train()
        for i, (images, labels) in enumerate(train_loader):

            # cast data examples to cuda or cpu device
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            weights = torch.ones(output.shape[1]).to(device)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss.item(), 'dataloss')
            #here can be logging
        model.eval()
        loss_eval = 0
        l = 0.
        for i, (images, labels) in enumerate(val_loader):
            output = model(images)

            loss_eval += criterion(output, labels).item()
            l += 1
        print('accuaracy: {}'.format(loss_eval/l))


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='unet hyperparameters')
    train()