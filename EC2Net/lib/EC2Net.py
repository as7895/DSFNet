import torch
import torch.nn as nn
import torch.nn.functional as F
# from .Res2Net_v1b import res2net50_v1b_26w_4s
from lib.ResNet import ResNet
import torchvision.models as models
from utils.tensor_ops import cus_sample, upsample_add
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

#........... Basic Convolution Layer...........
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#.........Parallel Separable Compact Enhancement (PSE)............
class PSE(nn.Module):
    def __init__(self, channels=64, r=4):
        super(PSE, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei

#............. Shallow Crossed-Context Aggregation Mechanism (SCCA)......
class SCCA(nn.Module):
    def __init__(self, channel=64):
        super(SCCA, self).__init__()

        self.msca = PSE()
        self.upsample = cus_sample
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x, y):

        y = self.upsample(y, scale_factor=1)
        xy = x + y
        wei = self.msca(xy)
        xo = x * wei + y * (1 - wei)
        xo = self.conv(xo)

        return xo



#................. Deep Cross-Context Aggregation (DCCA) ...........................
class DCCA(nn.Module):
    def __init__(self, channels):
        super(DCCA, self).__init__()
        self.convk1d1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.convk3d1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.convk5d1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.convk7d1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        # self.convk3d3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, groups=channels, bias=False)
        # self.convk3d5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, groups=channels, bias=False)
        # self.convk3d7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=7, dilation=7, groups=channels, bias=False)
        #self.convk1 = nn.Conv2d(channels, channels // 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, a):
        c1 = self.convk1d1(x)
        c2 = self.convk3d1(x)
        c3 = self.convk5d1(x)
        #c4 = self.convk7d1(x)
        #c5 = self.convk3d1(x)
        #c6 = self.convk3d5(x)
        #c7 = self.convk3d7(x)

        out = self.relu(x*a[0] + c1*a[1] + c2*a[2] + c3*a[3] )
        return out #self.convk1(out)

#............. Dual Cross-Fusion Mechanism (DCFM)..............
class DCFM(nn.Module):
    def __init__(self, channel):
        super(DCFM, self).__init__()
        self.reset_gate = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.update_gate = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.out_gate_1 = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.out_gate_2 = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        update = torch.sigmoid(self.update_gate(x))
        reset = torch.sigmoid(self.reset_gate(x))
        out1 = torch.tanh(self.out_gate_1(torch.cat([x1, x2 * reset], dim=1)))
        out2 = torch.tanh(self.out_gate_2(torch.cat([x2, x1 * reset], dim=1)))
        x = (x1 + x2) * (1 - update) + (out1 + out2) * update
        return x

#............... Main model..................
class EC2Net(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=64):
        super(EC2Net, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = ResNet()#pretrained=True)

        #.............Basic Convolution Layers.............
        self.conv1 = nn.Sequential(BasicConv2d(256, 64, kernel_size=3, stride=1, padding=1, relu=True),nn.BatchNorm2d(64),nn.ReLU())
        self.conv2 = nn.Sequential(BasicConv2d(512, 64, kernel_size=3, stride=1, padding=1, relu=True),nn.BatchNorm2d(64),nn.ReLU())
        self.conv3 = nn.Sequential(BasicConv2d(1024, 64, kernel_size=3, stride=1, padding=1, relu=True),nn.BatchNorm2d(64),nn.ReLU())
        self.conv4 = nn.Sequential(BasicConv2d(2048, 64, kernel_size=3, stride=1, padding=1, relu=True),nn.BatchNorm2d(64),nn.ReLU())
        #............. SCCA.............
        self.scca = SCCA()

        #............. DCCA..............
        self.dcca = DCCA(channel)
        self.pse1 = PSE()
        self.pse2 = PSE()

        #............. DCFM..............
        self.dcfm = DCFM(channel)
      #............. Classifier.............
        self.classifier = nn.Conv2d(64, 1, 1)
        self.upsample_add = upsample_add
        
        if self.training:
            self.initialize_weights()
        

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
        #........ convolution layer....................
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x2)
        conv3 = self.conv3(x3)
        conv4 = self.conv4(x4)
        #........... SCCA..............
        
        conv1 = F.interpolate(conv1, scale_factor=1, mode='bilinear', align_corners=False)
        conv2 = F.interpolate(conv2, scale_factor=2, mode='bilinear', align_corners=False)
    
        x_scca = self.scca(conv1,conv2)
      
        #....... PSE......................
       
        self.upsample = cus_sample

        x_pse1 = self.pse1(conv3)
        x_pse2 = self.pse2(self.upsample(conv4, scale_factor=2))
        
        #......... DCCA ...................
        x_dcca = self.dcca(x_pse1, x_pse2)

        #...... DCFM .........
        
        x_dcfm = self.dcfm(x_scca, self.upsample(x_dcca,scale_factor=4)) 

        #...... classifier.........
        s3 = self.classifier(x_dcfm)
        s3 = F.interpolate(s3, scale_factor=4, mode='bilinear', align_corners=False)

        return s3
        
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


if __name__ == '__main__':
    ras = EC2Net().cuda()
    input_tensor = torch.randn(2, 3, 352, 352).cuda()

    out = ras(input_tensor)
