import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.BCSANet import BCSANet
from torchvision.utils import make_grid
from tools.data import get_loader, test_dataset
from tools.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt

# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

#............ Training dataset.....
image_root = opt.rgb_root
gt_root = opt.gt_root
flow_root = opt.flow_root
#................ Testing dataset......
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_flow_root = opt.test_flow_root
save_path = opt.save_path

# build the model
model = BCSANet()

num_parms = 0

# if (opt.load is not None):
#     model.load_pre(opt.load)
#     print('load model from ', opt.load)


model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
pparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pparams)

# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, flow_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_flow_root, opt.trainsize)
total_step = len(train_loader)

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, flow) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            flow = flow.cuda()
            flow = flow.repeat(1, 3, 1, 1).cuda()
            s = model(images, flow)

            sal_loss = CE(s, gts)
            #sal_loss1 = CE(s1, gts)

            loss = sal_loss #+ sal_loss1
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'BCSANet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'BCSANet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, flow, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            flow = flow.cuda()
            flow = flow.repeat(1, 3, 1, 1).cuda()
            res= model(image, flow)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'BCSANet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    # start the training
    for epoch in range(1, opt.epoch):
        # if (epoch % 50 ==0 and epoch < 60):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)

        test(test_loader, model, epoch, save_path)