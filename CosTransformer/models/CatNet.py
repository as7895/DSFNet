""""
backbone is ResNet50
"""
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from models.ResNet import ResNet
import torch.nn.functional as F
import os
import onnx

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class CostTransformer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(CostTransformer, self).__init__()

        self.rgb_resnet = ResNet()
        self.flow_resnet = ResNet()

        self.afem = AFEM(2048, 2048)  #
        self.down1 = conv3x3_bn_relu(4096, 2048)
        self.down2 = conv3x3_bn_relu(2048, 1024)
        self.down3 = conv3x3_bn_relu(1024, 512)
        self.down4 = conv3x3_bn_relu(512, 256)

        self.cmfm1 = CMFM(2048, 12, 12, 4)  # 
        self.cmfm2 = CMFM(1024, 24, 24, 4)
        self.cmfm3 = CMFM(512, 48, 48, 4)
        self.cmfm4 = CMFM(256, 96, 96, 4)

        self.fgwa1 = FGWA()
        self.fgwa2 = FGWA()

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv256_32 = conv3x3_bn_relu(256, 64)
        self.conv64_1 = conv3x3(64, 1)

        self.relu = nn.ReLU(True)

    def forward(self, x, y):
        #........ RGB Information..........
        x = self.rgb_resnet.conv1(x)
        x = F.relu(self.rgb_resnet.bn1(x),inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        r4 = self.rgb_resnet.layer1(x)  # 256*96
        r3 = self.rgb_resnet.layer2(r4)  # 512*48
        r2 = self.rgb_resnet.layer3(r3)  # 1024*24
        r1 = self.rgb_resnet.layer4(r2)  # 2048*12
        r1 = self.afem(r1)
        #......... Motion Information........
        m = self.flow_resnet.conv1(y)
        m = F.relu(self.flow_resnet.bn1(m),inplace=True)
        m = F.max_pool2d(m, kernel_size=3, stride=2, padding=1)
        m4 = self.flow_resnet.layer1(m)  # 256*96
        m3 = self.flow_resnet.layer2(m4)  # 512*48
        m2 = self.flow_resnet.layer3(m3)  # 1024*24
        m1 = self.flow_resnet.layer4(m2)  # 2048*12
        m1 = self.afem(m1)
        # CMFM
        fuse1 = self.cmfm1(r1, m1)  # [2048]
        fuse1 = self.down1(fuse1)
        fuse2 = self.cmfm2(r2, m2)  # [1024]
        fuse2 = self.down2(fuse2)
        fuse3 = self.cmfm3(r3, m3)  # [512]
        fuse3 = self.down3(fuse3)
        fuse4 = self.cmfm4(r4, m4)  # [256]
        fuse4 = self.down4(fuse4)
        #....... Flow Guided Window Attention (FGWA)
        end_fuse1, out43, out432 = self.fgwa1(fuse1, fuse2, fuse3, fuse4)
        end_fuse = self.fgwa23(fuse1, out43, out432, end_fuse1, end_fuse1)

        end_sal = self.conv256_32(end_fuse)  # [b,32]
        end_sal1 = self.conv256_32(end_fuse1)
        
        out = self.up4(end_sal)
        out1 = self.up4(end_sal1)
        sal_out = self.conv64_1(out)
        sal_out1 = self.conv64_1(out1)

        return sal_out, sal_out1

#..... AFEM................
class AFEM(nn.Module):
    def __init__(self, dim, in_dim):
        super(AFEM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim),
                                       nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv3 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv4 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)

        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2 * out2 + conv2

        conv3 = self.conv3(x)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        conv4 = self.conv4(x)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:],
                           mode='bilinear')
        return self.fuse(torch.cat((conv1, out2, out3, out4, conv5), 1))

#.... DFEM..................
class DFEM(nn.Module):
    def __init__(self, infeature):
        super(DFEM, self).__init__()
        self.depth_spatial_attention = SpatialAttention()
        self.depth_channel_attention = ChannelAttention(infeature)
        self.rd_spatial_attention = SpatialAttention()

    def forward(self, r, d):
        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        d_f = d * sa
        d_f = d + d_f
        d_ca = self.depth_channel_attention(d_f)
        d_out = d * d_ca
        return d_out
#......... RFEM...........
class RFEM(nn.Module):
    def __init__(self, infeature, w=12, h=12, heads=4):
        super(RFEM, self).__init__()
        self.rgb_channel_attention = ChannelAttention(infeature)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()

    def forward(self, r, d):

        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        r_f = r + r_f
        r_ca = self.rgb_channel_attention(r_f)
        r_out = r * r_ca
        return r_out

#............. CMFM................
class CMFM(nn.Module):
    def __init__(self, infeature, w=12, h=12, heads=4):
        super(CMFM, self).__init__()
        self.dfem = DFEM(infeature)
        self.rfem = RFEM(infeature, w, h, heads)
        self.ca = ChannelAttention(infeature * 2)

    def forward(self, r, d):
        fr = self.rfem(r, d)
        fd = self.dfem(r, d)
        mul_fea = fr * fd
        add_fea = fr + fd
        fuse_fea = torch.cat([mul_fea, add_fea], dim=1)
        return fuse_fea

#.... MSFA......
class MSFA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSFA, self).__init__()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = conv3x3_bn_relu(in_ch, out_ch)
        self.aff = AFF(out_ch)

    def forward(self, fuse_high, fuse_low):
        fuse_high = self.up2(fuse_high)
        fuse_high = self.conv(fuse_high)
        fe_decode = self.aff(fuse_high, fuse_low)
    
        return fe_decode

# FGWA
class FGWA(nn.Module):
    def __init__(self):
        super(FGWA, self).__init__()
        self.cfm12 = MSFA(2048, 1024)
        self.cfm23 = MSFA(1024, 512)
        self.cfm34 = MSFA(512, 256)
        self.conv256_512 = conv3x3_bn_relu(256, 512)
        self.conv256_1024 = conv3x3_bn_relu(256, 1024)
        self.conv256_2048 = conv3x3_bn_relu(256, 2048)

    def forward(self, fuse4, fuse3, fuse2, fuse1, iter=None):
        if iter is not None:
            out_fuse4 = F.interpolate(iter, size=(12, 12), mode='bilinear')
            out_fuse4 = self.conv256_2048(out_fuse4)
            fuse4 = out_fuse4 + fuse4

            out_fuse3 = F.interpolate(iter, size=(24, 24), mode='bilinear')
            out_fuse3 = self.conv256_1024(out_fuse3)
            fuse3 = out_fuse3 + fuse3

            out_fuse2 = F.interpolate(iter, size=(48, 48), mode='bilinear')
            out_fuse2 = self.conv256_512(out_fuse2)
            fuse2 = out_fuse2 + fuse2

            fuse1 = iter + fuse1

            out43 = self.cfm12(fuse4, fuse3)
            out432 = self.cfm23(out43, fuse2)
            out4321 = self.cfm34(out432, fuse1)
            return out4321
        else:
            out43 = self.cfm12(fuse4, fuse3)  # [b,1024,24,24]
            out432 = self.cfm23(out43, fuse2)  # [b,512,48,48]
            out4321 = self.cfm34(out432, fuse1)  # [b,256,96,96]
            return out4321, out43, out432


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        # xa = x * residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = torchvision.transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    net = CostTransformer()
    a = torch.randn([2, 3, 384, 384])
    b = torch.randn([2, 3, 384, 384])
    s, e, s1 = net(a, b)
    print("s.shape:", e.shape)
