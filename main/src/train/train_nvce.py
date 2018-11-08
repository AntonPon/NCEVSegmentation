import torch

from main.src.models.unet_model import Unet
from main.src.models.nvce_model import NVCE
from main.data.cityscapes_loader import get_data_loader, decode_segmap
from main.src.unet.accuracy import runningScore
from main.src.loss.cross_entropy_loss import cross_entropy2d
from main.src.utils.augmentation import RandomRotate, RandomHorizontallyFlip, Compose
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def train(agrs=''):
    experiment_number = 1
    batch_szie = 6
    img_size = (256, 256)
    worker_num = 4
    cuda_usage = True
    epoch_number = 1000

    device = 'cpu'
    if torch.cuda.is_available() and cuda_usage:
        device = 'cuda:0'

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_path = os.path.join(save_dir_root, 'results', 'experiment_{}'.format(experiment_number))
    writer = SummaryWriter(log_dir=save_dir_path)

    nvce_model_loader = os.path.join('/home/anpon/master_thesis', 'unet_citiscapes_best_model_nvce.pkl')
    path_to_model = os.path.join(save_dir_root, 'unet_cityscapes_best_model_iou_3.pkl')

    # root_data_path = '/home/user/Documents/datasets/cityscapes'
    root_data_path = '/../../../data/anpon/cityscapes'
    root_data_path_add = '/../../../data/anpon/cityscapes2/leftImg8bit_sequence'

    transform = Compose([RandomRotate(10), RandomHorizontallyFlip()])
    val_loader, train_loader = get_data_loader(root_data_path, root_data_path_add, transform, img_size,
                                               batch_size=batch_szie, worker_num=worker_num)
    val_decode_segmap = decode_segmap

    unet = DataParallel(Unet())
    unet.to(device)
    if path_to_model is not None:
        if os.path.isfile(path_to_model):
            print("Loading model and optimizer from checkpoint '{}'".format(path_to_model))
            checkpoint = torch.load(path_to_model)
            unet.load_state_dict(checkpoint['model_state'])
        else:
            print("No checkpoint found at '{}'".format(path_to_model))

    model = NVCE(unet)  # Unet()
    model.to(device)

    if os.path.isfile(nvce_model_loader):
        checkpoint_nvce = torch.load(nvce_model_loader)
        model.load_state_dict(checkpoint_nvce['model_state'])
        print('model nvce has been uploaded')
    else:
        print('the nvce model path is not found')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=5e-4)
    # Setup Metrics
    criterion = cross_entropy2d
    running_metrics = runningScore(19)
    running_metrics_train = runningScore(19)

    len_trainload = len(train_loader)
    len_valload = len(val_loader)
    best_iou = -1
    for epoch in range(0, epoch_number):
        train_loss = 0.

        model.train()
        for i, (images, next_images, labels) in enumerate(train_loader):
            # cast data examples to cuda or cpu device
            next_images = next_images.to(device)
            images = images.to(device)
            labels = labels.to(device)
            model(next_images)

            output = model(images, is_keyframe=False)

            loss = criterion(input=output, target=labels, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_iou = output.data.max(1)[1].cpu().numpy()
            ground_truth = labels.data.cpu().numpy()
            running_metrics_train.update(ground_truth, output_iou)
            train_loss += loss.item()

        score_train, _ = running_metrics_train.get_scores()
        running_metrics_train.reset()
        writer.add_scalar('miou_train', score_train['Mean IoU : \t'], epoch)
        writer.add_scalar('epoch_loss_train', train_loss / len_trainload, epoch)

        val_loss = 0.
        model.eval()
        for i, (images, next_images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            next_images = next_images.to(device)

            model(next_images)
            output = model(images, is_keyframe=False)
            output_iou = output.data.max(1)[1].cpu().numpy()

            ground_truth = labels.data.cpu().numpy()
            running_metrics.update(ground_truth, output_iou)
            val_loss += criterion(input=output, target=labels, device=device).item()

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        writer.add_scalar('epoch_loss_val', val_loss / len_valload, epoch)
        writer.add_scalar('miou_val', score['Mean IoU : \t'], epoch)

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch + 1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state, "{}_{}_best_model_nvce.pkl".format('unet', 'cityscapes'))
    writer.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='unet hyperparameters')
    train()
