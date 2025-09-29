from torch import nn, optim
import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data import *
from net import *
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = r'weight'
data_path = r'dataset/train'

def dice_coefficient(pred, target, smooth=1e-10):
    pred = (pred >= 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target)

def save_checkpoint(epoch, model, optimizer):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, f'{weight_path}/{epoch}.pth')
    print(f"checkpoint saved at {epoch}.pth")

def multi_stage_lr(epoch):
    if epoch < 10:
        return 1.0
    elif epoch < 20:
        return 0.5
    elif epoch < 30:
        return 0.25
    else:
        return 0.3125

# =================== training script ===================
if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)
    net = FAD_Net().to(device)

    # define optimizer and scheduler
    opt = optim.Adam(
        net.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=multi_stage_lr)
    loss_fun = MSELoss()

    # ========== checkpoint branch ==========
    checkpoint_flag = False           # set True to load checkpoint
    checkpoint_name = ''           # specify pth name (without suffix)
    start_epoch = 1                  # default start from 1

    if checkpoint_flag:
        checkpoint = torch.load(f'{weight_path}/{checkpoint_name}.pth')
        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"checkpoint {checkpoint_name}.pth loaded, resume from epoch {start_epoch}")

    # ========== training loop ==========
    total_train_dice = 0.0
    total_epoch_count = 0
    max_epoch = 400

    for epoch in range(start_epoch, max_epoch + 1):
        total_dice = 0.0
        total_batches = 0

        for i, (image, segment_image, segment_name) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)

            train_loss = loss_fun(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            dice = dice_coefficient(out_image, segment_image)
            total_dice += dice.item()
            total_batches += 1

            if i % 100 == 0:
                print(f'Epoch {epoch} -- Batch {i} -- Loss: {train_loss.item():.6f} -- Dice: {dice.item():.6f}')

        average_dice = total_dice / total_batches if total_batches > 0 else 0.0
        total_train_dice += average_dice
        total_epoch_count += 1

        print(f'Epoch {epoch} - Average Dice: {average_dice:.6f} - LR: {opt.param_groups[0]["lr"]:.6f}')

        if epoch % 5 == 0:
            save_checkpoint(epoch, net, opt)

        scheduler.step()

    final_average_dice = total_train_dice / total_epoch_count if total_epoch_count > 0 else 0.0
    print(f'Training Finished - Average Dice: {final_average_dice:.6f}')