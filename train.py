from torch import nn, optim
import torch
import os
from torch.utils.data import DataLoader
from data import *
from torchvision.utils import save_image
from net import *
device=torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')     # todo cuda:0, cuda:1
weight_path=r'weight'
data_path=r'dataset/train'
save_path=r'train_image'

def dice_coefficient(pred, target, smooth=1e-10):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def val(data):
    for j in range(image.size(0)):
        output_image_name = segment_name[j]  # 使用原始文件名
        output_image_path = os.path.join(save_path, output_image_name)  # 生成完整路径
        save_image(out_image[j], output_image_path)  # 保存输出图像


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target)

def save_checkpoint(epoch, model, optimizer):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, f'weight/{epoch}.pth')

# 训练脚本
if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
    net=FAD_Net().to(device)
    chekpoint = False
    chek=''
    if chekpoint:
        net.load_state_dict(torch.load(f'weight/{chek}.pth'))    #用于加载之前保存的模型参数，并将其赋给当前模型，帮助模型恢复到之前的训练状态，从而可以继续训练或直接用于推理
        print('successful loading weight')

    opt = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    loss_fun = MSELoss()  # 使用交叉熵损失函数
    total_train_dice = 0.0
    total_epoch_count = 0  # 用于记录总 epoch 数

    epoch=1
    while (epoch<=800):
        total_dice = 0.0
        total_batches = 0
        for i,(image,segment_image,segment_name) in enumerate(data_loader):
            image, segment_image,=image.to(device),segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            # 计算 Dice 损失
            dice = dice_coefficient(out_image, segment_image)
            # 累加当前 epoch 的 Dice 系数和批次数
            total_dice += dice.item()
            total_batches += 1
            #

            if i%100==0:
                print(f'{epoch}--{i}--train_loss==>{train_loss.item()}--dice={dice.item()}')

        if (epoch % 5) == 0:
            save_checkpoint(epoch,net, opt)
            torch.save(net.state_dict(), f'weight/{epoch}.pth')

        # 计算并打印当前 epoch 的平均 Dice 系数
        average_dice = total_dice / total_batches if total_batches > 0 else 0.0
        print(f'Epoch {epoch} - Average Dice: {average_dice}')

        scheduler.step()  # 更新学习率
        epoch+=1

    # 计算并打印整个训练过程的平均 Dice 系数
    final_average_dice = total_train_dice / total_epoch_count if total_epoch_count > 0 else 0.0
    print(f'Training Finished - 平均dice= {final_average_dice}')
