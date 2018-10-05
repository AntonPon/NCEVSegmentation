import torch

from matplotlib import pyplot as plt
from main.src.unet.unet_model import Unet
from main.data.data_loader_implemented import get_data_loader, decode_segmap
from main.src.unet.accuracy import runningScore
from main.src.loss.cross_entropy_loss import cross_entropy2d
from main.src.utils.augmentation import RandomRotate, RandomHorizontallyFlip, Compose


def train(agrs=''):

    batch_szie = 6
    img_size = (256, 256)
    worker_num = 8
    cuda_usage = True

    root_data_path = '/home/user/Documents/datasets/cityscapes'

    transform = Compose([RandomRotate(10), RandomHorizontallyFlip()])
    val_loader, train_loader = get_data_loader(root_data_path, transform, img_size, batch_size=batch_szie,
                                               worker_num=worker_num)

    val_decode_segmap = decode_segmap

    device = 'cpu'
    if torch.cuda.is_available() and cuda_usage:
        device = 'cuda:1'
    model = Unet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=5e-4)
    criterion = cross_entropy2d
    # Setup Metrics
    running_metrics = runningScore(19)

    for epoch in range(0, 100):
        model.train()
        for i, (images, labels) in enumerate(train_loader):

            # cast data examples to cuda or cpu device
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = criterion(input=output, target=labels, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, 100, loss.item()))
        # here can be logging
        model.eval()
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images).data.cpu().numpy()
            ground_truth = labels.data.cpu().numpy()

            running_metrics.update(ground_truth, output)
            if epoch % 1 == 0:
                # if i_val == 0:
                #    for row in gt[0][507:511]:
                #        print(row[100:250])
                for i_v in range(ground_truth.shape[0]):
                    plt.subplot(121)
                    plt.imshow(val_decode_segmap(output[i_v]))
                    plt.xlabel('predicted: {}_{}'.format(epoch, i))
                    plt.subplot(122)
                    plt.imshow(val_decode_segmap(ground_truth[i_v]))
                    plt.xlabel('ground_truth: {}_{}'.format(epoch, i))
                    plt.savefig('images/epoch:{}_{}.png'.format(epoch, i))
                plt.close()

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch + 1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state, "{}_{}_best_model_iou.pkl".format('unet', 'citiscapes'))

if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='unet hyperparameters')
    train()
