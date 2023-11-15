import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from data import get_loader
from utils import AvgMeter, update_predict
from model.PACNet import PACNet
# import IOU

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20, help='epoch number')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=6, help='batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='input size')
    parser.add_argument('--trainset', type=str, default='PACNet')
    parser.add_argument('--train_type', type=str, default='finetune', help='finetune or pretrain_rgb or pretrain_flow')
    opt = parser.parse_args()

    # build models
    model = PACNet().cuda()

    if opt.train_type == 'finetune':
        save_path = '/content/drive/MyDrive/snapshots/{}/'.format(opt.trainset)
        # ---- data preparing ----
        src_dir = '/content/drive/MyDrive/data/DAVIS/Train'
        image_root = src_dir + '/RGB/'
        #flow_root = src_dir + '/depth/'
        gt_root = src_dir + '/GT/'

        train_loader = get_loader(image_root=image_root, gt_root=gt_root,
                                        batchsize=opt.batchsize, trainsize=opt.trainsize, shuffle=True,
                                        num_workers=4, pin_memory=True)
        total_step = len(train_loader)
        #
        #update_predict(model)
    elif opt.train_type == 'pretrain_rgb':
        save_path = '../snapshot/{}_rgb/'.format(opt.trainset)
        # ---- data preparing ----
        src_dir = './data/TrainSet_StaticAndVideo'
        image_root = src_dir + '/Imgs/'
        gt_root = src_dir + '/GTs/'

        train_loader = get_loader(image_root=image_root,  gt_root=gt_root,
                                        batchsize=opt.batchsize, trainsize=opt.trainsize, shuffle=True,
                                        num_workers=4, pin_memory=True)
        total_step = len(train_loader)
    elif opt.train_type == 'pretrain_flow':
        save_path = '../snapshot/{}_flow/'.format(opt.trainset)
        # ---- data preparing ----
        src_dir = './dataset/TrainSet_Video'
        flow_root = src_dir + '/Flow/'
        gt_root = src_dir + '/ground-truth/'

        train_loader = get_loader(image_root=flow_root, gt_root=gt_root,
                                        batchsize=opt.batchsize, trainsize=opt.trainsize, shuffle=True,
                                        num_workers=4, pin_memory=True)
        total_step = len(train_loader)
    else:
        raise AttributeError('No train_type: {}'.format(opt.train_type))

    # ---- parallel model ----
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    params = model.parameters()
    
    optimizer = torch.optim.SGD(params, opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    BCE = torch.nn.BCEWithLogitsLoss()
    def structure_loss(pred, mask):
      weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
      wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
      wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

      pred  = torch.sigmoid(pred)
      inter = ((pred*mask)*weit).sum(dim=(2,3))
      union = ((pred+mask)*weit).sum(dim=(2,3))
      wiou  = 1-(inter+1)/(union-inter+1)
      return (wbce+wiou).mean()

    
    tparams = sum(param.numel() for param in model.parameters())   
    print(tparams)
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]

    # training
    for epoch in range(0, opt.epoch):
        scheduler.step()
        model.train()
        loss_record = AvgMeter()

        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- get data ----
                images,  gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- multi-scale training ----
                trainsize = int(round(opt.trainsize*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                s = model(images)
                loss1 = structure_loss(s, gts) 
                loss = loss1 
                # ---- backward ----
                loss.backward()
                optimizer.step()
                # ---- show loss ----
                if rate == 1:
                    loss_record.update(loss.data, opt.batchsize)
                                      
            if i % 25 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

        os.makedirs(save_path, exist_ok=True)
        if epoch > 15:
            if (epoch+1) % 1 == 0:
                torch.save(model.state_dict(), save_path + opt.trainset + '-{}epoch.pth'.format(epoch))
                print('[Model Saved] Path: {}'.format(save_path + opt.trainset + '-{}epoch.pth'.format(epoch)))


if __name__ == '__main__':
    main()