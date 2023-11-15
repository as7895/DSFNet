import torch
import torch.nn as nn
import torch.nn.functional as F
from SDFM import (SDFM, DenseTransLayer,)
from MDEM import DFM
from models.BaseBlocks import BasicConv_PRelu
import torchvision
from TCAM import BCSA
from SDFM import (SDFM, DenseTransLayer,)
from tensor_ops import cus_sample, upsample_add

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

### BCSANet small ###
class BCSANet(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(BCSANet,self).__init__()
        #...... Layer1...........
        self.stage1 = RSU7(in_ch,16,512)
        self.bcsa1 = BCSA(512,512)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        #...... Layer2...........
        self.stage2 = RSU6(512,16,512)
        self.bcsa2 = BCSA(512,512)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        #...... Layer3...........
        self.stage3 = RSU5(512,16,512)
        self.bcsa3 = BCSA(512,512)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        #...... Layer4...........
        self.stage4 = RSU4(512,16,512)
        self.bcsa4 = BCSA(512,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        #...... Layer5...........
        self.stage5 = RSU4F(512,16,512)
        self.bcsa5 = BCSA(512,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        #...... Layer6...........
        self.stage6 = RSU4F(512,16,512)

        # decoder
        #...... Decoder1...........
        self.stage5d = RSU4F(1024,16,512)
        self.sdfm1 = SDFM(512, 512, 512, 3, 4)
        #...... Decoder2...........
        self.stage4d = RSU4(1024,16,512)
        self.sdfm2 = SDFM(512, 512, 512, 3, 4)
        #...... Decoder3...........
        self.stage3d = RSU5(1024,16,512)
        self.sdfm3 = SDFM(512, 512, 512, 3, 4)
        #...... Decoder4...........
        self.stage2d = RSU6(1024,16,512)
        self.sdfm4 = SDFM(512, 512, 512, 3, 4)
        #...... Decoder5...........
        self.stage1d = RSU7(1024,16,512)
        self.sdfm5 = SDFM(512, 512, 512, 3, 4)
        self.side1 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)
        self.upsample = cus_sample
        self.up_add = upsample_add
        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)
        self.transfer = DenseTransLayer(6, 6)
    def forward(self,x, y):

        hx = x
        hy = y
        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hy1 = self.stage1(hy)
        hy = self.pool12(hy1)
        hxy_1 = self.bcsa1(hx, hy)
        #stage 2
        hx2 = self.stage2(hxy_1)
        hx = self.pool23(hx2)
        hy2 = self.stage2(hxy_1)
        hy = self.pool23(hy2)
        hxy_2 = self.bcsa2(hx, hy)
        #stage 3
        hx3 = self.stage3(hxy_2)
        hx = self.pool34(hx3)
        hy3 = self.stage3(hxy_2)
        hy = self.pool34(hy3)
        hxy_3 = self.bcsa3(hx, hy)
        #stage 4
        hx4 = self.stage4(hxy_3)
        hx = self.pool45(hx4)
        hy4 = self.stage4(hxy_3)
        hy = self.pool45(hy4)
        hxy_4 = self.bcsa4(hx, hy)
        #stage 5
        hx5 = self.stage5(hxy_4)
        hx = self.pool56(hx5)
        hy5 = self.stage5(hxy_4)
        hx = self.pool56(hy5)
        hx = self.upsample(hx, scale_factor=2)
        #print(hx.size())
        #print(hy.size())
        hxy_5 = self.bcsa5(hx, hy)
        #stage 6
        hx6_r = self.stage6(hxy_5)
        hx6up_r = _upsample_like(hx6_r, hxy_5)
        hx6_m = self.stage6(hxy_5)
        hx6up_m = _upsample_like(hx6_m, hxy_5)
        
        #decoder
        hx5d_r = self.stage5d(torch.cat((hx6up_r,hx5),1))
        hx5dup_r = _upsample_like(hx5d_r,hx4)

        hx5d_m = self.stage5d(torch.cat((hx6up_m,hy5),1))
        hx5dup_m = _upsample_like(hx5d_m,hy4)

        hx4d_r = self.stage4d(torch.cat((hx5dup_r,hx4),1))
        hx4dup_r = _upsample_like(hx4d_r,hx3)
        
        hx4d_m = self.stage4d(torch.cat((hx5dup_m,hy4),1))
        hx4dup_m = _upsample_like(hx4d_m,hy3)

        hx3d_r = self.stage3d(torch.cat((hx4dup_r,hx3),1))
        hx3dup_r = _upsample_like(hx3d_r,hx2)

        hx3d_m = self.stage3d(torch.cat((hx4dup_m,hy3),1))
        hx3dup_m = _upsample_like(hx3d_m,hy2)

        hx2d_r = self.stage2d(torch.cat((hx3dup_r,hx2),1))
        hx2dup_r = _upsample_like(hx2d_r,hx1)

        hx2d_m = self.stage2d(torch.cat((hx3dup_m,hy2),1))
        hx2dup_m = _upsample_like(hx2d_m,hy1)

        hx1d_r = self.stage1d(torch.cat((hx2dup_r,hx1),1))
        hx1d_m = self.stage1d(torch.cat((hx2dup_m,hy1),1))


        #side output
        d1_r = self.side1(hx1d_r)
        d1_m = self.side1(hx1d_m)
        #....... side output 5...........
        d2_r = self.side2(hx2d_r)
        d2_m = self.side2(hx2d_r)
        d2_r = _upsample_like(d2_r,d1_r)
        d2_m = _upsample_like(d2_m,d1_m)
        #....... side output 2 ...........
        d3_r = self.side3(hx3d_r)
        d3_m = self.side3(hx3d_m)
        d3_r = _upsample_like(d3_r,d1_r)
        d3_m = _upsample_like(d3_m,d1_m)
        #....... side output 3...........
        d4_r = self.side4(hx4d_r)
        d4_r = _upsample_like(d4_r,d1_r)
        d4_m = self.side4(hx4d_m)
        d4_m = _upsample_like(d4_m,d1_m)
        #....... side output 4...........
        d5_r = self.side5(hx5d_r)
        d5_r = _upsample_like(d5_r,d1_r)
        d5_m = self.side5(hx5d_m)
        d5_m = _upsample_like(d5_m,d1_m)
        #....... side output 5...........
        d6_r = self.side6(hx6_r)
        d6_r = _upsample_like(d6_r,d1_r)
        d6_m = self.side6(hx6_m)
        d6_m = _upsample_like(d6_m,d1_m)
        #........... Combine decoder output
        d0_r = self.up_add(torch.cat((d1_r,d2_r,d3_r,d4_r,d5_r,d6_r),1))
        d0_m = self.up_add(torch.cat((d1_m,d2_m,d3_m,d4_m,d5_m,d6_m),1))
        #........... DenseLayer......
        out = self.transfer(d0_r, d0_m)
        out = self.outconv(out)
        #print(out.size())
        return F.sigmoid(out)#, F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)