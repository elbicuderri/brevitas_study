from brevitas.core import bit_width
from brevitas.nn.mixin import act
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, BatchNorm2dToQuantScaleBias, QuantAvgPool2d, QuantLinear
from brevitas.core.quant import QuantType

def ConvBlock(
    in_channels,
     out_channels,
      kernel_size,
       padding,
        stride,
          weight_bit,
           activation_bit,
            bias=False,
             **kwargs):
    block = nn.Sequential(
        QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, weight_quant=weight_bit, bias=bias),
        BatchNorm2dToQuantScaleBias(num_features=out_channels, eps=1e-5, weight_quant=weight_bit, bias_quant=weight_bit),
        QuantReLU(bit_width=activation_bit),
        QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, weight_quant=weight_bit, bias=bias),
        BatchNorm2dToQuantScaleBias(num_features=out_channels, eps=1e-5, weight_quant=weight_bit, bias_quant=weight_bit),
    )
    return block

class SimpleResNet(Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()

        # self.identity = QuantIdentity(act_quant=8)

        self.relu = QuantReLU(bit_width=8)

        self.conv0 = nn.Sequential(
            QuantConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False, weight_quant=4),
            BatchNorm2dToQuantScaleBias(16, eps=12-5, weight_quant=4, bias_quant=4),
            QuantReLU(act_quant=8)
        )

        self.block11 = ConvBlock(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, weight_bit=4, activation_bit=8)

        self.block12 = ConvBlock(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, weight_bit=4, activation_bit=8)

        self.conv2 = QuantConv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, weight_quant=4, bias=False)

        self.block21 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, weight_bit=4, activation_bit=8)

        self.block22 = ConvBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, weight_bit=4, activation_bit=8)

        self.conv3 = QuantConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, weight_quant=4, bias=False)

        self.block31 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, weight_bit=4, activation_bit=8)

        self.block32 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, weight_bit=4, activation_bit=8)
        
        self.avg_pool = QuantAvgPool2d(8, trunc_quant=8)
        self.fc = QuantLinear(64, 10, weight_quant=4, bias_quant=4)

    def forward(self, x):
        batch = x.size(0)

        out0 = self.conv0(x)

        # res11 = self.identity(out0)
        res11 = out0

        out11 = self.block11(out0)
        out11 += res11
        out11 = self.relu(out11)

        # res12 = self.identity(out11)
        res12 = out11

        out12 = self.block12(out11)
        out12 += res12
        out12 = self.relu(out12)

        res21 = self.conv2(out12)
        out21 = self.block21(out12)
        out21 += res21
        out21 = self.relu(out21)

        # res22 = self.identity(out21)
        res22 = out21

        out22 = self.block22(out21)
        out22 += res22
        out22 = self.relu(out22)

        res31 = self.conv3(out22)
        out31 = self.block31(out22)
        out31 += res31
        out31 = self.relu(out31)

        # res32 = self.identity(out31)
        res32 = out31

        out32 = self.block32(out31)
        out32 += res32
        out32 = self.relu(out32)

        out4 = self.avg_pool(out32)
        out4 = out4.view(batch, -1)
        out = self.fc(out4)

        return out