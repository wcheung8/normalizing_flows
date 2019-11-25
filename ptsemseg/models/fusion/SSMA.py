import torch
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.segnet import segnet
from ptsemseg.models.fusion.deeplab import DeepLab
from ptsemseg.models.fusion.decoder import build_decoder
from ptsemseg.models.utils import *
import torchvision.models as models


class SSMA(nn.Module):
    def __init__(self, backbone='segnet', output_stride=16, n_classes=21,
                 sync_bn=True, freeze_bn=False):

        super(SSMA, self).__init__()

        if backbone == 'segnet':
            self.expert_A = segnet(n_classes=n_classes, in_channels=3, is_unpooling=True)
            self.expert_B = segnet(n_classes=n_classes, in_channels=3, is_unpooling=True)
            vgg16 = models.vgg16(pretrained=True)
            self.expert_A.init_vgg16_params(vgg16)
            self.expert_B.init_vgg16_params(vgg16)
            self.SSMA_skip1 = _SSMABlock(24, 4)
            self.SSMA_skip2 = _SSMABlock(24, 4)
            self.SSMA_ASPP = _SSMABlock(512, 4)
        else:
            self.expert_A = DeepLab(backbone, output_stride, n_classes, sync_bn, freeze_bn)
            self.expert_B = DeepLab(backbone, output_stride, n_classes, sync_bn, freeze_bn)
            self.SSMA_skip1 = _SSMABlock(64, 4)
            self.SSMA_skip2 = _SSMABlock(512, 4)
            self.SSMA_ASPP = _SSMABlock(512, 4)

        self.decoder = _Decoder(n_classes, in_channels=512)

    def forward(self, input):

        A, A_llf1, A_llf2, A_aspp = self.expert_A.forward_SSMA(input[:, :3, :, :])
        B, B_llf1, B_llf2, B_aspp = self.expert_B.forward_SSMA(input[:, 3:, :, :])

        fused_ASPP = self.SSMA_ASPP(A_aspp, A_aspp)
        fused_skip1 = self.SSMA_skip1(A_llf1, B_llf1)
        fused_skip2 = self.SSMA_skip2(A_llf2, B_llf2)

        x = self.decoder(fused_ASPP, fused_skip1, fused_skip2)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


class _SSMABlock(nn.Module):
    def __init__(self, n_channels, compression_rate=6):
        super(_SSMABlock, self).__init__()
        self.bottleneck = nn.Conv2d(
            2 * n_channels,
            n_channels // compression_rate,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )

        self.gate = nn.Conv2d(
            n_channels // compression_rate,
            2 * n_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )

        conv_mod = nn.Conv2d(
            2 * n_channels,
            n_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )

        self.fuser = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_channels)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, B):
        AB = torch.cat([A, B], dim=1)

        G = self.bottleneck(AB)
        G = F.relu(G)
        G = self.gate(G)
        G = self.sigmoid(G)

        AB = AB * G

        fused = self.fuser(AB)

        return fused


class _Decoder(nn.Module):
    def __init__(self, n_classes=11, in_channels=512, compress_size=24):
        super(_Decoder, self).__init__()
        
        
        self.leg1 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(in_channels))
                                  
        self.compress1 = nn.Sequential(nn.AvgPool2d(1),
                                       conv2DBatchNormRelu(in_channels,compress_size,1,1,0))
                                       
                                       
        self.leg2 = nn.Sequential(conv2DBatchNormRelu(in_channels + compress_size, in_channels, 3, 1, 1),
                                  conv2DBatchNormRelu(in_channels, in_channels, 3, 1, 1),
                                  nn.ConvTranspose2d(in_channels, in_channels, 8, stride=4, padding=2),
                                  nn.BatchNorm2d(in_channels))
                                  
      
        self.compress2 = nn.Sequential(nn.AvgPool2d(1),
                                       conv2DBatchNormRelu(in_channels,compress_size,1,1,0))
                                       
       
        self.leg3 = nn.Sequential(conv2DBatchNormRelu(in_channels + compress_size,in_channels,3,1,1),
                                  conv2DBatchNormRelu(in_channels,in_channels,3,1,1),
                                  conv2DBatchNormRelu(in_channels,in_channels,3,1,1),
                                  nn.ConvTranspose2d(in_channels, n_classes, 8, stride=4, padding=2),
                                  nn.BatchNorm2d(n_classes))
        

    def forward(self, ASSP, skip1, skip2):
        
        x = self.leg1(ASSP)

        features1 = self.compress1(x) * skip2
        x = torch.cat([x, features1], dim=1)
        x = self.leg2(x)

        features2 = self.compress2(x) * skip1
        x = torch.cat([x, features2], dim=1)
        x = self.leg3(x)

        return x

if __name__ == "__main__":
    model = SSMA(backbone='mobilenet', output_stride=16, n_classes=11)
    model.eval()
    input = torch.rand(2, 6, 512, 512)
    output = model(input)
    print(output.size())
