import torch

#from main.src.models.unet_model import Unet
#from main.src.models.nvce_model import NVCE

from main.src.models.fpn_model import FPN
from main.src.models.nvce_fpn_model import NVCE_FPN
from main.data.cityscapes_loader import get_data_loader, decode_segmap
from main.src.train.accuracy import runningScore
from main.src.loss.cross_entropy_loss import cross_entropy2d
from main.src.utils.augmentation import RandomRotate, RandomHorizontallyFlip, Compose
from tensorboardX import SummaryWriter
import os
import sys

print('python version is: {}'.format(sys.version))


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def train(agrs=''):

    batch_szie = 3
    img_size = (512, 512)
    worker_num = 1
    cuda_usage = True
    epoch_number = 1000
    alpha = 0.3
    distance_type = 'triple_random_detach'
    experiment_number = 'fpn_with_loss_{}_distance_{}'.format('random_triple_loss_1layer', distance_type)
    device = 'cpu'
    if torch.cuda.is_available() and cuda_usage:
        device = 'cuda:0'

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_path = os.path.join(save_dir_root, 'results', 'experiment_{}'.format(experiment_number))
    writer = SummaryWriter(log_dir=save_dir_path)

    nvce_model_loader = os.path.join('/home/anpon/master_thesis', 'fpn_loss_cityscapes_best_model_nvce.pkl')
    path_to_model = os.path.join('/home/anpon/master_thesis/fpn_bold_rewrite_cityscapes_best_model_iou.pkl')  # (save_dir_root, 'unet_cityscapes_best_model_iou_3.pkl')

    # root_data_path = '/home/user/Documents/datasets/cityscapes'
    root_data_path = '/../../../data/anpon/cityscapes'
    root_data_path_add = '/../../../data/anpon/cityscapes2/leftImg8bit_sequence'

    transform = Compose([RandomRotate(10), RandomHorizontallyFlip()])
    val_loader, train_loader = get_data_loader(root_data_path, root_data_path_add, transform, img_size,
                                               batch_size=batch_szie, worker_num=worker_num)
    #val_decode_segmap = decode_segmap

    pre_trained = FPN(num_classes=19) #DataParallel(Unet())
    pre_trained.to(device)
    if path_to_model is not None:
        if os.path.isfile(path_to_model):
            print("Loading model and optimizer from checkpoint '{}'".format(path_to_model))
            checkpoint = torch.load(path_to_model)
            pre_trained.load_state_dict(checkpoint['model_state'])
        else:
            print("No checkpoint found at '{}'".format(path_to_model))

    model = NVCE_FPN(pre_trained) #NVCE(pre_trained)  # Unet()
    model.to(device)

    if os.path.isfile(nvce_model_loader):
        checkpoint_nvce = torch.load(nvce_model_loader)
        model.load_state_dict(checkpoint_nvce['model_state'])
        print('model nvce has been uploaded')
    else:
        print('the nvce model path is not found')
    # model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=5e-4)
    # Setup Metrics
    criterion = cross_entropy2d
    running_metrics = runningScore(19)
    running_metrics_keyframe = runningScore(19)
    running_metrics_train = runningScore(19)
    running_metrics_keyframe_train = runningScore(19)

    len_trainload = len(train_loader)
    len_valload = len(val_loader)
    best_iou = -1

    #for param in pre_trained.parameters():
    #    param.requires_grad = False

    for epoch in range(0, epoch_number):
        model.train()
        pre_trained.eval()
        train_loss = 0.
        for i, (images, prev_images, labels, dst) in enumerate(train_loader):
            # cast data examples to cuda or cpu device
            prev_images = prev_images.to(device)
            images = images.to(device)
            labels = labels.to(device)

            output_key = model(images)
            output_key_fpn = pre_trained(prev_images).data.max(1)[1]

            output_prev = model(prev_images)
            output = model(images, is_keyframe=False)

            # alpha = 0.4#get_alpha(dst)
            loss = alpha * criterion(input=output, target=labels, device=device) + \
                   (1 - alpha)/2 * (criterion(input=output_key, target=labels, device=device) +
                                  criterion(input=output_prev, target=output_key_fpn, device=device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_iou = output.data.max(1)[1].cpu().numpy()
            output_iou_key = output_key.data.max(1)[1].cpu().numpy()

            ground_truth = labels.data.cpu().numpy()

            running_metrics_train.update(ground_truth, output_iou)
            running_metrics_keyframe_train.update(ground_truth, output_iou_key)
            train_loss += loss.item()

        score_train, _ = running_metrics_train.get_scores()
        score_key_train, _ = running_metrics_keyframe_train.get_scores()
        running_metrics_train.reset()
        running_metrics_keyframe_train.reset()

        writer.add_scalar('miou/train_current_frame', score_train['Mean IoU : \t'], epoch)
        writer.add_scalar('miou/train_key_frame', score_key_train['Mean IoU : \t'], epoch)
        writer.add_scalar('total_loss/train', train_loss / len_trainload, epoch)

        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for i, (images, prev_images, labels, dst) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                prev_images = prev_images.to(device)

                output_key = model(images)
                model(prev_images)
                output = model(images, is_keyframe=False)

                output_iou = output.data.max(1)[1].cpu().numpy()
                output_iou_key = output_key.data.max(1)[1].cpu().numpy()

                ground_truth = labels.data.cpu().numpy()
                running_metrics.update(ground_truth, output_iou)
                running_metrics_keyframe.update(ground_truth, output_iou_key)

                #alpha = get_alpha(dst)
            # val_loss += criterion(input=output, target=labels, device=device).item()
                val_loss += alpha * criterion(input=output, target=labels, device=device).item() + \
                            (1 - alpha) * criterion(input=output_key, target=labels, device=device).item()

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        score_key, _ = running_metrics_keyframe.get_scores()
        running_metrics.reset()
        running_metrics_keyframe.reset()

        writer.add_scalar('total_loss/val', val_loss / len_valload, epoch)
        writer.add_scalar('miou/val_current_frame', score['Mean IoU : \t'], epoch)
        writer.add_scalar('miou/val_key_frame', score_key['Mean IoU : \t'], epoch)

        if score_key['Mean IoU : \t'] >= best_iou:
        #if ((1 - alpha) * score_key_train['Mean IoU : \t'] + alpha * score_train['Mean IoU : \t']) >= best_iou:
            #best_iou = score['Mean IoU : \t']
            best_iou = (1 - alpha) * score_key_train['Mean IoU : \t'] + alpha * score_train['Mean IoU : \t']
            state = {'epoch': epoch + 1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state, "../../../data/anpon/snapshots_masterth/{}_{}_alpha_{}_distance_{}_model_nvce.pkl".format('fpn_loss', 'cityscapes', '0_3', distance_type))
    writer.close()


def get_alpha(current_dist):
    a = 1.
    return a / (current_dist + 1)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='train hyperparameters')
    train()
