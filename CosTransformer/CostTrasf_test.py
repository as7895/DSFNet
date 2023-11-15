import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.CatNet import CostTransformer
from tools.data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/content/drive/MyDrive/dataset/DAVIS/Test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = CostTransformer()
model.load_state_dict(torch.load('/media/omnisky/sdc/rp/Swin1/cpts3/CATNet_epoch_best.pth'))
model.cuda()
model.eval()
# test

test_datasets = ['DAVIS','FBMS','SegTrack-V2','MCL','DAVSOD','DAVSOD-Difficult']
#............ Test dataset
for dataset in test_datasets:
    save_path = './final/train_2985/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(edge_save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    flow_root = dataset_path + dataset + '/Flow/'
    test_loader = test_dataset(image_root, gt_root, flow_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, flow, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        flow = flow = flow.repeat(1,3,1,1).cuda()
        res, res1= model(image, flow)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        # ndarray to image
        cv2.imwrite(save_path + name, res*255)
    print('Test Done!')
