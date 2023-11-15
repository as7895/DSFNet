import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from lib.ResNet import ResNet
from tensor_ops import cus_sample, upsample_add

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#................. Fully Connected Layer (FC) .....................
class FC(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(FC, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu 
        self.linear = nn.Conv2d(in_channels, out_channels,1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        y = self.linear(x)
        y = self.bn(y)
        y = F.relu(y)
        return y
#....................... Importance Perception Fusion (IPF) ......................        
class IPF(nn.Module):
    def __init__(self, channel):
        super(IPF, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(11)
        self.mlp = FC(channel, channel, True, True)
        self.upsample = cus_sample
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = self.gap(x)
        y = self.gap(y) 
        xy = x + y
        wei = self.mlp(xy)
        xo = x * wei + y * (1 - wei)
        xo = self.conv(xo)

        return xo

# Gather Diffusion Reinforcement (GDR) module
class GDR(nn.Module):
    # GDR
    def __init__(self, in_channel, out_channel):
        super(GDR, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

#.......................... Crossmodality Refinement and Complement (CRC) module..........
class CRC(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(CRC, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.reduc_1 = nn.Sequential(BasicConv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(BasicConv2d(in_dim, out_dim, kernel_size=1), act_fn)
        
        self.layer_10 = BasicConv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = BasicConv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        
        self.layer_11 = nn.Sequential(BasicConv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_21 = nn.Sequential(BasicConv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        

        self.layer_ful1 = nn.Sequential(BasicConv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
       
    def forward(self, rgb, depth):
        
        ################################
        x_rgb = self.reduc_1(rgb)
        x_dep = self.reduc_2(depth)
        
        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)
        
        rgb_w = nn.Sigmoid()(x_rgb1)
        dep_w = nn.Sigmoid()(x_dep1)
        
        ##
        x_rgb_w = x_rgb.mul(dep_w)
        x_dep_w = x_dep.mul(rgb_w)
        
        x_rgb_r = x_rgb_w + x_rgb
        x_dep_r = x_dep_w + x_dep
        
        ## fusion 
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)
        
        
        ful_mul = torch.mul(x_rgb_r, x_dep_r)         
        x_in1   = torch.reshape(x_rgb_r,[x_rgb_r.shape[0],1,x_rgb_r.shape[1],x_rgb_r.shape[2],x_rgb_r.shape[3]])
        x_in2   = torch.reshape(x_dep_r,[x_dep_r.shape[0],1,x_dep_r.shape[1],x_dep_r.shape[2],x_dep_r.shape[3]])
        x_cat   = torch.cat((x_in1, x_in2),dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul,ful_max),dim=1)
        
        out1 = self.layer_ful1(ful_out)
         
        return out1

###############################################################################

class PSNet(nn.Module):
    def __init__(self, ind=50):
        super(PSNet, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.upsample_2 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=11, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
       
        #Backbone model

        self.layer_rgb  = ResNet()
        self.layer_dep  = ResNet()
        
        self.layer_dep0 = BasicConv2d(1, 3, kernel_size=1)

        # funsion encoders # 

        self.fu_1 = CRC(64,64)
        self.pool_fu_1 = maxpool()
        
        self.fu_2 = CRC(64, 64)
        self.pool_fu_2 = maxpool()
        
        self.fu_3 = CRC(64, 64)
        self.pool_fu_3 = maxpool()

        self.fu_4 = CRC(64, 64)
        self.pool_fu_4 = maxpool()
        #......................... GDR............

        self.gdr1 = GDR(350,350)
        self.gdr2 = GDR(512,512)
        self.gdr3 = GDR(512,512)
        self.gdr4 = GDR(512,512)
        #.................. Dimensionality reduction.....
       
        self.d2 = BasicConv2d(256, 512, 3, 1,1)
        self.d3 = BasicConv2d(512, 512, 3, 1,1)
        self.d4 = BasicConv2d(1024, 512, 3, 1,1)
        self.d5 = BasicConv2d(2048, 512, 3, 1,1)

        # # decoders #       
        self.down = BasicConv2d(512, 350, 3, 1,1)
        self.down2 = BasicConv2d(350, 64, 3, 1, 1)
        self.down1 = BasicConv2d(512, 64, 3, 1,1)
        #............ IPF.......
        self.ful_layer4 = IPF(64)

        self.upsample = cus_sample
        #............... classification of object.......
        self.classification = nn.Conv2d(64, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, imgs, motions):
        
        img_0 = self.layer_rgb.conv1(imgs)
        img_0 = self.layer_rgb.bn1(img_0)
        img_0 = self.layer_rgb.relu(img_0)
        img_0 = self.layer_rgb.maxpool(img_0)
        img_1 = self.layer_rgb.layer1(img_0)#..256
        img_2 = self.layer_rgb.layer2(img_1)#..512
        img_3 = self.layer_rgb.layer3(img_2)#..1024
        img_4 = self.layer_rgb.layer4(img_3)#..2048
        img_1 = self.d2(img_1)
        img_2 = self.d3(img_2)
        img_3 = self.d4(img_3)
        img_4 = self.d5(img_4)

        #............... motion......
        mot_0 = self.layer_dep.conv1(self.layer_dep0(motions))
        mot_0 = self.layer_dep.bn1(mot_0)
        mot_0 = self.layer_dep.relu(mot_0)
        mot_0 = self.layer_dep.maxpool(mot_0)
        mot_1 = self.layer_dep.layer1(mot_0)#..256
        mot_2 = self.layer_dep.layer2(mot_1)#..512
        mot_3 = self.layer_dep.layer3(mot_2)#..1024
        mot_4 = self.layer_dep.layer4(mot_3)#..2048
        #........... Dimension reduction...........

        mot_1 = self.d2(mot_1)
        mot_2 = self.d3(mot_2)
        mot_3 = self.d4(mot_3)
        mot_4 = self.d5(mot_4)

        #--------------GDR---------------#
        img_1 = self.down(img_1)
        gdr_1 = self.gdr1(img_1)
        gdr_1 = self.down2(gdr_1)
        gdr_2 = self.gdr2(img_2)
        gdr_2 = self.down1(gdr_2)
        gdr_3 = self.gdr3(img_3)
        gdr_3 = self.down1(gdr_3)
        gdr_4 = self.gdr4(img_4)
        gdr_4 = self.down1(gdr_4)

        #------------ CRC (RGB).............................
       
        ful_1    = self.fu_1(gdr_1, self.upsample(self.down1(img_4),scale_factor=8))
        ful_2    = self.fu_2(self.upsample(gdr_2, scale_factor=2), ful_1)
        ful_3    = self.fu_3(self.upsample(gdr_3, scale_factor=4), ful_2)
        ful_out1 = self.fu_4(self.upsample(gdr_4, scale_factor=8), ful_3)
        
        #.................. CRC (Motion) ....................
  
        ful_1    = self.fu_1(gdr_1, self.upsample(self.down1(mot_4),scale_factor=8))
        ful_2    = self.fu_2(self.upsample(gdr_2, scale_factor=2), ful_1)
        ful_3    = self.fu_3(self.upsample(gdr_3, scale_factor=4), ful_2)
        ful_out2 = self.fu_4(self.upsample(gdr_4, scale_factor=8), ful_3)
      
        #............ IPF...................
        x_ful_42 = self.ful_layer4(ful_out1, ful_out2)
        
        #........ Classification .................
        ful_out1 = self.classification(ful_out1)
        ful_out1 = self.upsample_3(ful_out1)
        
        ful_out2 = self.classification(ful_out2)
        ful_out2 = self.upsample_3(ful_out2)
        
        x_ful_42 = self.classification(x_ful_42)
        ful_out42 = self.upsample_2(x_ful_42)
        
        return self.sigmoid(ful_out1),self.sigmoid(ful_out2), self.sigmoid(ful_out42)
    
     