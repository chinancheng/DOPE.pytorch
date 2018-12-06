import argparse
import os
from config import Config
from network.rtpose_vgg import get_model, use_vgg
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter


def _train(path_to_data_dir, path_to_logs_dir, batch_size, epochs, restore):
    writer = SummaryWriter(path_to_logs_dir)
    dataset = Dataset(path_to_data=path_to_data_dir)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
    model = get_model(trunk='vgg19')
    model = model.cuda()
    use_vgg(model, './model', 'vgg19')
    if restore:
        model.load_state_dict(torch.load(restore))

    model.train()

    for i in range(20):
        for param in model.model0[i].parameters():
            param.requires_grad = False
    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_vars, lr=0.0001)

    epoch = 0
    step = 1
    while epoch != epochs:
        for batch_index, (images, heatmaps_target, pafs_target) in enumerate(dataloader):
            images = images.cuda()
            _, saved_for_loss = model(images)
            # loss = _loss(saved_for_loss, heatmaps_target.cuda(), pafs_target.cuda())/batch_size
            loss = _loss(saved_for_loss, heatmaps_target.cuda(), pafs_target.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print('Epoch: {}, Step: {}, Loss: {}' .format(epoch, step, loss.data.item()))
            writer.add_scalar('train/loss', loss, step)
            step += 1
        epoch += 1
        print('save checkpoint')
        torch.save(model.state_dict(), os.path.join(path_to_logs_dir, 'checkpoint.pth'))


def _validate():
    return None


def _loss(saved_for_loss, heatmaps_target, pafs_target):
    # criterion = nn.MSELoss(reduction='sum').cuda()
    criterion = nn.MSELoss().cuda()
    loss = 0
    for i in range(Config.output_stage):
        pafs_pred = saved_for_loss[2*i]
        heatmaps_pred = saved_for_loss[2*i+1]
        heatmaps_loss = criterion(heatmaps_pred, heatmaps_target)
        pafs_loss = criterion(pafs_pred, pafs_target)
        loss += heatmaps_loss + pafs_loss

    return loss


if __name__ == '__main__':
    def main(args):
        path_to_data_dir = args.path_to_data_dir
        path_to_logs_dir = args.path_to_logs_dir
        os.makedirs(path_to_logs_dir, exist_ok=True)
        batch_size = args.batch_size
        epochs = args.epochs
        restore = args.restore_checkpoint
        _train(path_to_data_dir, path_to_logs_dir, batch_size, epochs, restore)


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path_to_data_dir', default='./data/BottleDataSet/JockMiddle', help='path to data directory')
    parser.add_argument('-l', '--path_to_logs_dir', default='./logs-v1', help='path to logs directory')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=50, type=int, help='epochs')
    parser.add_argument('-r', '--restore_checkpoint', default=None,
                     help='path to restore checkpoint file, e.g., ./logs/model-100.pth')
    main(parser.parse_args())