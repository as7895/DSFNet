import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from models.CASNet import CASNet
from utils.data import get_loader
import torch.nn.functional as F
from utils.utils import clip_gradient, poly_lr, AvgMeter
import torch.nn.functional as F
#from utils.AdaX import AdaXW
from thop import profile


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record3 = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images.float()).cuda()
            gts = Variable(gts.float()).cuda()
            # ---- rescale ----
            #trainsize = int(round(opt.trainsize*rate/32)*32)
            #if rate != 1:
            #    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            #    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_3 = model(images)
            # ---- loss function ----
            loss3 = F.binary_cross_entropy_with_logits(lateral_map_3, gts)
            #loss2 = F.binary_cross_entropy_with_logits(s1, gts)
            #loss1 = F.binary_cross_entropy_with_logits(s2, gts)
            # ---- backward ----
            loss = loss3#+loss2+loss3
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            macs, params = profile(model, inputs=(images, ))
            
            print(macs)
            print(params)
            # ---- recording loss ----
            #if rate == 1:
            loss_record3.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-3: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.show()))
    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'C2FNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'C2FNet-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=25, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=384, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='/content/drive/MyDrive/dataset/DAVIS/Train', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='CASNet')
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = CASNet().cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    # print total parameters
    para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameter:',para)
    
    # load the dataset
    print('........ Load the dataset ........')
    image_root = '{}/RGB/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    

    print(".......... Start Training .........")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)
