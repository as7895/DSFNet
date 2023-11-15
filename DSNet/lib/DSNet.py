import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from lib.ResNet import ResNet
from utils.utils import cus_sample, upsample_add
def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)


class DepthBranch(nn.Module):
    def __init__(self, in_channels):
        super(DepthBranch, self).__init__()
        self.bottleneck1 = DWConv( in_channels, in_channels, stride=1)
        self.bottleneck2 = DWConv( in_channels, in_channels, stride=1)
        self.bottleneck3 = DWConv( in_channels, in_channels, stride=1)
        self.bottleneck4 = DWConv( in_channels, in_channels, stride=1)
        self.bottleneck5 = DWConv( in_channels, in_channels, stride=1)

        # self.conv_s_d = _ConvBNReLU(320,1,1,1)

        # nn.Sequential(_DSConv(c3, c3 // 4),
        #                           nn.Conv2d(c3 // 4, 1, 1), )

    def forward(self, x):
        #size = x.size()[2:]
        #feat = []

        x1 = self.bottleneck1(x)
        x2 = self.bottleneck2(x1)
        x3 = self.bottleneck3(x2)
        x4 = self.bottleneck4(x3)
        x5 = self.bottleneck5(x4)
        # s_d = self.conv_s_d(x5)

        #feat.append(x1)
        #feat.append(x2)
        ##feat.append(x3)
        #feat.append(x4)
        #feat.append(x5)
        return x5
#............................. Channel Attention (CA) ........
class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
#............................. DARP............
class DARP(nn.Module):
    def __init__(self, in_channels):
        super(DARP, self).__init__()
        self.d1 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d2 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d3 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d4 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d5 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d6 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.sa = SA()
        self.ca = CA(in_channels)
        
    def forward(self, x):
        size = x.size()[2:]
        feat = []

        x1 = self.d1(x)
        x11 = self.ca(x1)
        x1 = x1 * x11
        x2 = self.d2(x1)
        x22 = self.ca(x2)
        x2 = x2 * x22
        x3 = self.d3(x2)
        x33 = self.ca(x3)
        x3 = x3 * x33
        x4 = self.d4(x3)
        x44 = self.sa(x4)
        x4 = x4 * x44
        x5 = self.d5(x4)
        x55 = self.sa(x5)
        x5 = x5 * x55
        x6 = self.d6(x5)
        x66 = self.ca(x6)
        x6 = x6 * x66
        feat = x6
        #....... 
        # feat.append(x1)
        # feat.append(x2)
        # feat.append(x3)
        # feat.append(x4)
        # feat.append(x5)
        # feat.append(x6)
        return feat
#......................... SCNet...............
class SCNet(nn.Module):
    """Separable convolution Network(SCNet)"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(SCNet, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = DSConv(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = DSConv(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = DSConv(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = DSConv(in_channels, inter_channels, 1, **kwargs)
        self.out = DSConv(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(ConvBNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

def _make_layer( block, inplanes, planes, blocks, t=6, stride=1):
    layers = []
    layers.append(block(inplanes, planes, t, stride))
    for i in range(1, blocks):
        layers.append(block(planes, planes, t, 1))
    return nn.Sequential(*layers)

class DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            ConvBNReLU(in_channels, in_channels , 1),
            ConvBNSig(in_channels, in_channels , stride),
            DSConv(in_channels, in_channels , stride),

            # dw
            DWConv(in_channels , in_channels , stride),
            # pw-linear
            nn.Conv2d(in_channels , out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

class DCNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DCNet, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
        #self.depthbranch = DepthBranch(in_channels)
    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        #x = self.depthbranch(x)
        return x


class DSNet(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, padding=1):
        super(DSNet, self).__init__()
        self.resnet = ResNet()
        self.down1 = SCNet(256, in_channels, kernel_size=3, padding=1)
        self.down2 = SCNet(512, in_channels, kernel_size=3, padding=1)
        self.down3 = SCNet(1024, in_channels, kernel_size=3, padding=1)
        self.down4 = SCNet(2048, in_channels, kernel_size=3, padding=1)
        self.down = SCNet(128,in_channels, kernel_size=3, padding=padding)
        # self.down = SCNet(128,in_channels, kernel_size=3, padding=padding)
        # self.down = SCNet(128,in_channels, kernel_size=3, padding=padding)

        self.conv1_1 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1_2 = SCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.drap1 = DARP(in_channels)
        self.maxpool1 = nn.MaxPool2d(1)

        self.conv2_1 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2_2 = SCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.drap2 = DARP(in_channels)
        self.maxpool2 = nn.MaxPool2d(1)

        self.conv3_1 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3_2 = SCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.drap3 = DARP(in_channels)
        self.maxpool3 = nn.MaxPool2d(1)

        self.conv4_1 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv4_2 = SCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.drap4 = DARP(in_channels)
        self.maxpool4 = nn.MaxPool2d(1)

        self.conv5_1 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5_2 = SCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)

        self.conv6_1 = SCNet(2*in_channels, in_channels, kernel_size=3, padding=1)
        self.conv6_2 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv6_t = nn.ConvTranspose2d(in_channels, in_channels, 1, stride=1)

        self.conv7_1 = SCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv7_2 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv7_t = nn.ConvTranspose2d(in_channels, in_channels, 1, stride=1)

        self.conv8_1 = SCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv8_2 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv8_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)

        self.conv9_1 = SCNet(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv9_2 = DCNet(in_channels, in_channels, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.upsample = cus_sample
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x =  F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #print(x.size())
        x1 = self.resnet.layer1(x)
        
        x2 = self.resnet.layer2(x1)
        
        x3 = self.resnet.layer3(x2)
        #x3 = self.down3(x)
        #print(x3.size())
        x4 = self.resnet.layer4(x3)
        #............ Downsampling......
        x1 = self.down1(x1)
        #print(x1.size())
        x2 = self.down2(x2)
        #print(x2.size())
        x3 = self.down3(x3)
        #print(x3.size())
        x4 = self.down4(x4)
        #print(x4.size())
        #......... First Encoder................
        conv1 = F.relu(self.conv1_1(x1))
        conv1 = F.relu(self.conv1_2(conv1))
        conv1 = self.drap1(conv1)
        pool1 = self.maxpool1(conv1)
       #............ Second Encoder............
        conv2 = F.relu(self.conv2_1(x2))
        conv2 = F.relu(self.conv2_2(conv2))
        conv2 = self.drap2(conv2)
        pool2 = self.maxpool2(conv2)
       #............ Third Encoder...............
        conv3 = F.relu(self.conv3_1(x3))
        conv3 = F.relu(self.conv3_2(conv3))
        conv3 = self.drap3(conv3)
        pool3 = self.maxpool3(conv3)
       #............. Fourth Encoder..............
        conv4 = F.relu(self.conv4_1(x4))
        conv4 = F.relu(self.conv4_2(conv4))
        conv4 = self.drap4(conv4)
        #print(conv4.size())
        pool4 = self.maxpool4(conv4)
        #print(pool4.size())
       #........... Latent Feature................
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))
        #print(conv5.size())
        #print(conv4.size())
       #........... fourt Decoder...................
        conv4 = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=False)
        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        #up6 = self.down(up6)
        #print(up6.size())
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))
       #............ Third Decoder...................
        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        up7 = self.down(up7)
        #print(up7.size())
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))
       #............. Second Decoder...................
        conv7 = F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=False)
        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        up8 = self.down(up8)
        #print(up8.size())
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))
       #............ First Decoder......................
        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        up9 = self.down(up9)
        #print(up9.size())
        conv9 = F.relu(self.conv9_1(up9))
        conv9 = F.relu(self.conv9_2(conv9))
       #...... Classifier...................
        conv10 = self.conv10(conv9)
        conv10 = cus_sample(conv10, scale_factor=4)#F.interpolate(conv9, scale_factor=4, mode='bilinear', align_corners=False)
        out = F.sigmoid(conv10)
        return out

    