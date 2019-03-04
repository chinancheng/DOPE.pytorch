import argparse
import os
from config import Config
from network.rtpose_vgg import get_model, use_vgg
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import numpy as np
from tensorboardX import SummaryWriter


def _train(path_to_data_dir, path_to_logs_dir, batch_size, epochs, restore):
    writer = SummaryWriter(path_to_logs_dir)
    dataset = Dataset(path_to_data=path_to_data_dir)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataset = Dataset(path_to_data=path_to_data_dir, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0, drop_last=True)
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
    best_mse = 1.0
    while epoch != epochs:
        for batch_index, (images, heatmaps_target, pafs_target) in enumerate(dataloader):
            images = images.cuda()
            _, saved_for_loss = model(images)
            loss, heatmaps_losses, pafs_losses = _loss(saved_for_loss, heatmaps_target.cuda(), pafs_target.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print('Epoch: {}, Step: {}, Loss: {}' .format(epoch, step, loss.data.item()))
            writer.add_scalar('train_total_loss/loss', loss, step)
            for stage, (heatmaps_loss, pafs_loss) in enumerate(zip(heatmaps_losses, pafs_losses)):
                writer.add_scalar('train_heatmaps_loss/stage_{}' .format(str(stage)), heatmaps_loss, step)
                writer.add_scalar('train_pafs_loss/stage_{}' .format(str(stage)), pafs_loss, step)
            if step % 1000 == 0:
                pafs_loss, heatmaps_loss = _validate(model, val_dataloader)
                total_loss = pafs_loss + heatmaps_loss
                print('Validation Paf MSE: {} Heatmap MSE: {} Total MSE: {}' .format(pafs_loss, heatmaps_loss, total_loss))
                writer.add_scalar('val/heatmaps_loss', heatmaps_loss, step)
                writer.add_scalar('val/pafs_loss', pafs_loss, step)
                writer.add_scalar('val/total_loss', total_loss, step)
                if total_loss < best_mse:
                    print('Save checkpoint')
                    torch.save(model.state_dict(), os.path.join(path_to_logs_dir, 'checkpoint-best.pth'))
                    best_mse = total_loss
                    print('Best MSE: {}' .format(total_loss))
                model.train()
            step += 1
        epoch += 1
    print('Save checkpoint')
    torch.save(model.state_dict(), os.path.join(path_to_logs_dir, 'checkpoint-last.pth'))


def _validate(model, val_dataloader):
    model.eval()
    total_pafs_mse = []
    total_heatmaps_mse = []
    for batch_index, (images, heatmaps_target, pafs_target) in enumerate(val_dataloader):
        images = images.cuda()
        out, _ = model(images)
        pafs_pred = out[0].detach().cpu().numpy()
        pafs_target = pafs_target.numpy()
        heatmaps_pred = out[1].detach().cpu().numpy()
        heatmaps_target = heatmaps_target.numpy()
        pafs_mse = np.nanmean((pafs_pred - pafs_target) **2)
        heatmaps_mse = np.nanmean((heatmaps_pred - heatmaps_target) **2)
        total_pafs_mse.append(pafs_mse)
        total_heatmaps_mse.append(heatmaps_mse)
    total_pafs_mse = np.array(total_pafs_mse).mean()
    total_heatmaps_mse = np.array(total_heatmaps_mse).mean()

    return total_pafs_mse, total_heatmaps_mse


def _loss(saved_for_loss, heatmaps_target, pafs_target):
    criterion = nn.MSELoss().cuda()
    total_loss = 0
    heatmaps_losses = []
    pafs_losses = []
    for i in range(Config.output_stage):
        pafs_pred = saved_for_loss[2*i]
        heatmaps_pred = saved_for_loss[2*i+1]
        heatmaps_loss = criterion(heatmaps_pred, heatmaps_target)
        heatmaps_losses.append(heatmaps_loss)
        pafs_loss = criterion(pafs_pred, pafs_target)
        pafs_losses.append(pafs_loss)
        total_loss += heatmaps_loss + pafs_loss

    return total_loss, heatmaps_losses, pafs_losses


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
    parser.add_argument('-d', '--path_to_data_dir', default='/media/external/Bottle_dataset_split', help='path to data directory')
    parser.add_argument('-l', '--path_to_logs_dir', default='./logs/logs-dr-pr-jose-1', help='path to logs directory')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='epochs')
    parser.add_argument('-r', '--restore_checkpoint', default=None,
                     help='path to restore checkpoint file, e.g., ./logs/model-100.pth')
    main(parser.parse_args())
