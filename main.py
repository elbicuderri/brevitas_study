import torch
import torch.nn as nn
from quan_model.quan_resnet import SimpleResNet
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear
from brevitas.core.quant import QuantType
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleResNet().to(device=device)

summary(model, input_size=(3, 32, 32))