import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from models.SwinT import SwinTransformer
import torch.nn as nn
import onnx
import os
import math
from ResNet import ResNet
import torch.nn.functional as F
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

class Attention_unit(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_unit, self).__init__()
        # Three attention units, basic architecture of MDAB
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.spatial_attention3 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.channel_attention1 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention2 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention3 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

    def forward(self, g, x):
        # spa and cha attention at original scale
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        spa1 = F.relu(g1 + x1, inplace=True)
        cha1 = F.sigmoid(F.avg_pool2d(self.channel_attention1(spa1), spa1.size(2)))
        spa1 = F.sigmoid(self.spatial_attention1(spa1))

        # spa and cha attention at scale pooled by 3x3
        x2 = F.avg_pool2d(x, (3, 3))
        g2 = F.avg_pool2d(g, (3, 3))
        g2 = self.W_g(g2)
        x2 = self.W_x(x2)
        spa2 = F.relu( g2 + x2, inplace=True)
        cha2 = F.sigmoid(F.avg_pool2d(self.channel_attention2(spa2), spa2.size(2)))
        spa2 = F.sigmoid(self.spatial_attention2(spa2))
        spa2 = F.upsample(spa2, size=spa1.size()[2:], mode='bilinear', align_corners=True)
        cha2 = F.upsample(cha2, size=cha1.size()[2:], mode='bilinear', align_corners=True)

        # spa and cha attention at scale pooled by 6x6
        x3 = F.avg_pool2d(x, (6, 6))
        g3 = F.avg_pool2d(g, (6, 6))
        g3 = self.W_g(g3)
        x3 = self.W_x(x3)
        spa3 = F.relu( g3 + x3, inplace=True)
        cha3 = F.sigmoid(F.avg_pool2d(self.channel_attention3(spa3), spa3.size(2)))
        spa3 = F.sigmoid(self.spatial_attention3(spa3))
        spa3 = F.upsample(spa3, size=spa1.size()[2:], mode='bilinear', align_corners=True)
        cha3 = F.upsample(cha3, size=cha1.size()[2:], mode='bilinear', align_corners=True)
        #multi-scale attentive feature
        out = (x * spa1 + x * spa2 + x * spa3 + x * cha1 + x * cha2 + x * cha3)
        return out

    # def initialize(self):
    #     weight_init(self)

class Multi_Attention_unit(nn.Module):
    def __init__(self, F_k, F_g, F_l, F_int):
        super(Multi_Attention_unit, self).__init__()
        # Attention Units for three inputs, basic architecture for MBAB
        self.W_k = nn.Sequential(
            nn.Conv2d(F_k, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.spatial_attention3 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention1 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention2 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.channel_attention3 = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

    def forward(self, k, g, x):
        # spa and cha attention original scale
        k1 = self.W_k(k)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        spa1 = F.relu(k1 + g1 + x1,inplace=True)
        cha1 = F.sigmoid(F.avg_pool2d(self.channel_attention1(spa1), spa1.size(2)))
        spa1 = F.sigmoid(self.spatial_attention1(spa1))
        # spa and cha attention at scale pooled by 3x3
        x2 = F.avg_pool2d(x, (3, 3))
        g2 = F.avg_pool2d(g, (3, 3))
        k2 = F.avg_pool2d(k, (3, 3))
        k2 = self.W_k(k2)
        g2 = self.W_g(g2)
        x2 = self.W_x(x2)
        spa2 = F.relu(k2 + g2 + x2, inplace=True)
        cha2 = F.sigmoid(F.avg_pool2d(self.channel_attention2(spa2), spa2.size(2)))
        spa2 = F.sigmoid(self.spatial_attention2(spa2))
        spa2 = F.upsample(spa2,size=spa1.size()[2:],mode='bilinear',align_corners=True)
        cha2 = F.upsample(cha2, size=cha1.size()[2:], mode='bilinear', align_corners=True)
        # spa and cha attention at scale pooled by 6x6
        x3 = F.avg_pool2d(x, (6, 6))
        g3 = F.avg_pool2d(g, (6, 6))
        k3 = F.avg_pool2d(k, (6, 6))
        k3 = self.W_k(k3)
        g3 = self.W_g(g3)
        x3 = self.W_x(x3)
        spa3 = F.relu(k3 + g3 + x3, inplace=True)
        cha3 = F.sigmoid(F.avg_pool2d(self.channel_attention3(spa3), spa3.size(2)))
        spa3 = F.sigmoid(self.spatial_attention3(spa3))
        spa3 = F.upsample(spa3, size=spa1.size()[2:], mode='bilinear', align_corners=True)
        cha3 = F.upsample(cha3, size=cha1.size()[2:], mode='bilinear', align_corners=True)
        #Multi-scale attentive feature or detail flow
        out = (x * spa1 + x * spa2 + x * spa3 + x * cha1 + x * cha2 + x * cha3)
        return out

class CASNet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(CASNet, self).__init__()

        self.rgb_resnet = ResNet()
        self.afem = AFEM(256, 256)  # 
        
        self.cmfm1 = Multi_Attention_unit(2048, 2048, 2048, 2048)
        self.cmfm2 = Multi_Attention_unit(1024, 1024, 1024, 1024)
        self.cmfm3 = Multi_Attention_unit(512, 512, 512, 512)
        self.cmfm4 = Multi_Attention_unit(256, 256, 256, 256)

        self.decoder = Decoder()
        self.decoder2 = Decoder()

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv256_32 = conv3x3_bn_relu(256, 64)
        self.conv64_1 = conv3x3(64, 64)

        self.edge_layer = Edge_Module()
        self.edge_feature = conv3x3_bn_relu(1, 32)
        self.fuse_edge_sal = conv3x3(32, 1)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            conv3x3(64, 1)
        )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.rgb_resnet.conv1(x)
        x = F.relu(self.rgb_resnet.bn1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        r1 = self.rgb_resnet.layer1(x)
        r2 = self.rgb_resnet.layer2(r1)
        r3 = self.rgb_resnet.layer3(r2)
        r4 = self.rgb_resnet.layer4(r3)

        r1 = self.afem(r1)

        # 融合特征
        fuse1 = self.cmfm1(r4, r4, r4)  # [2048]
        fuse2 = self.cmfm2(r3, r3, r3)  # [1024]
        fuse3 = self.cmfm3(r2, r2, r2)  # [512]
        fuse4 = self.cmfm4(r1, r1, r1)  # [256]
        end_fuse1, out43, out432 = self.decoder(fuse1, fuse2, fuse3, fuse4)
        end_fuse = self.decoder2(fuse1, out43, out432, end_fuse1, end_fuse1)

        end_sal = self.conv256_32(end_fuse)  # [b,32]

        sal_out = self.up_edge(self.conv64_1(end_sal))

        return sal_out 


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
        # fe_decode = fuse_high + fuse_low
        return fe_decode

# Cascaded Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
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
    net = CASNet()
    a = torch.randn([2, 3, 384, 384])
    s, e, s1 = net(a)
    print("s.shape:", e.shape)
