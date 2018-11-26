import torch


import os
from main.src.models.fpn_model import FPN
from main.src.models.nvce_model import NVCE
from main.data.data_loader_implemented import get_data_loader, decode_segmap
from main.src.train.accuracy import runningScore
from main.src.loss.cross_entropy_loss import cross_entropy2d
from main.src.utils.augmentation import RandomRotate, RandomHorizontallyFlip, Compose, RandomCrop
from main.src.utils.util import add_info, save_model

from tensorboardX import SummaryWriter


def train(agrs=''):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    batch_szie = 8
    img_size = (512, 512)
    worker_num = 2
    cuda_usage = True
    epoch_number = 1000
    experiment_number = 'fpn_3'
    #root_data_path = '/home/user/Documents/datasets/cityscapes'
    root_data_path = '/../../../data/anpon/cityscapes'
    model_name = 'fpn_bold_rewrite_plus'

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_path = os.path.join(save_dir_root, 'results', 'experiment_{}'.format(experiment_number))
    writer = SummaryWriter(log_dir=save_dir_path)

    transform = Compose([RandomRotate(10), RandomHorizontallyFlip()])
    val_loader, train_loader = get_data_loader(root_data_path, transform, img_size, batch_size=batch_szie,
                                               worker_num=worker_num)

    #val_decode_segmap = decode_segmap

    train_data_len = len(train_loader)
    val_data_len = len(val_loader)

    device = 'cpu'
    if torch.cuda.is_available() and cuda_usage:
        device = 'cuda:0'
    model = FPN(num_classes=19)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=5e-4)
    criterion = cross_entropy2d
    # Setup Metrics
    running_metrics_val = runningScore(19)
    running_metrics_train = runningScore(19)


    best_iou = -1
    for epoch in range(0, epoch_number):
        #.train()
        #train_loss = 0.

        train_loss, running_metrics_train = train_net(train_loader, model, device, running_metrics_train, criterion, optimizer)
        '''
        for i, (images, labels) in enumerate(train_loader):

            # cast data examples to cuda or cpu device
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = criterion(input=output, target=labels, device=device)

            running_metrics_train.update(labels.data.cpu().numpy(), output.data.max(1)[1].cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        '''
        score_train, _ = running_metrics_train.get_scores()
        running_metrics_train.reset()
        print('adding info about train process')
        add_info(writer, epoch, train_loss/train_data_len, score_train['Mean IoU : \t'])

        #model.eval()
        #val_loss = 0.
        '''
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(input=output, target=labels, device=device)

            output = output.data.max(1)[1].cpu().numpy()
            ground_truth = labels.data.cpu().numpy()
            val_loss += loss.item()

            running_metrics_val.update(ground_truth, output)
        '''

        val_loss, running_metrics_val = val_net(val_loader, model, device, running_metrics_val, criterion)

        score, class_iou = running_metrics_val.get_scores()
        running_metrics_val.reset()
        add_info(writer, epoch, loss=val_loss/val_data_len,  miou=score['Mean IoU : \t'], mode='val')
        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            save_model(epoch, model.state_dict(), optimizer.state_dict(), model_name)


def train_net(train_loader, model,  device, metrics, criterion, optimizer):
    model.train()
    train_loss = 0.

    for i, (images, labels) in enumerate(train_loader):
        # cast data examples to cuda or cpu device
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(input=output, target=labels, device=device)

        metrics.update(labels.data.cpu().numpy(), output.data.max(1)[1].cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return (train_loss, metrics)


def val_net(val_loader, model,  device, metrics, criterion):
    model.eval()
    val_loss = 0.

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(input=output, target=labels, device=device)

            output = output.data.max(1)[1].cpu().numpy()
            ground_truth = labels.data.cpu().numpy()
            val_loss += loss.item()

            metrics.update(ground_truth, output)
    return (val_loss, metrics)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='train hyperparameters')
    train()

