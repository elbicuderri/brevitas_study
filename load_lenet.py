from brevitas.nn.quant_layer import QuantNonLinearActLayer
import torch
from torch.nn import Module
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear
from brevitas.core.quant import QuantType
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from quan_model.lenet_quan import QuantLeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# class QuantLeNet(Module):
#     def __init__(self):
#         super(QuantLeNet, self).__init__()
#         self.quant_inp = QuantIdentity(bit_width=8)
#         self.conv1 = QuantConv2d(3, 6, 5, weight_bit_width=4)
#         self.relu1 = QuantReLU(bit_width=8)
#         self.conv2 = QuantConv2d(6, 16, 5, weight_bit_width=4)
#         self.relu2 = QuantReLU(bit_width=8)
#         self.fc1   = QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4)
#         self.relu3 = QuantReLU(bit_width=8)
#         self.fc2   = QuantLinear(120, 84, bias=True, weight_bit_width=4)
#         self.relu4 = QuantReLU(bit_width=8)
#         self.fc3   = QuantLinear(84, 10, bias=False, weight_bit_width=4)

#     def forward(self, x):
#         out = self.quant_inp(x)
#         out = self.relu1(self.conv1(x))
#         out = F.max_pool2d(out, 2)
#         out = self.relu2(self.conv2(out))
#         out = F.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = self.relu3(self.fc1(out))
#         out = self.relu4(self.fc2(out))
#         out = self.fc3(out)
#         return out

model = QuantLeNet()

model.load_state_dict(torch.load("weight/quantlenet.pt"))
model.eval()

# for n, w in model.named_parameters():
#     print(n, ": ", w, "\n")

transform = transforms.Compose([
 transforms.ToTensor(), # 0 ~ 1
 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
]) # output[channel] = (input[channel] - mean[channel]) / std[channel]

valid_dataset = datasets.CIFAR10(root='~/data/cifar10/test/',
                                            train=False, 
                                            download=True,
                                            transform=transform)

valid_loader = DataLoader(dataset=valid_dataset,
                                          batch_size=1,
                                          shuffle=False)


val_num_correct = 0
val_num_samples = 0

for val_img, val_label in valid_loader:
    with torch.no_grad():
        val_img = val_img
        val_label = val_label
        val_output = model(val_img)

        _, val_predictions = val_output.max(1)
        val_num_correct += (val_predictions == val_label).sum() ## cpu??
        val_num_samples += val_predictions.size(0)


print(f"accuracy: {(val_num_correct / val_num_samples) * 100}")

# ## Error message
# core/function_wrapper/clamp.py", line 53, in forward
# return tensor_clamp(x, min_val=min_val, max_val=max_val)
# File "/home/seunghwan/anaconda3/envs/torch-1.7.0/lib/python3.8/site-packages/brevitas/function/ops.py", line 64, in tensor_clamp
# out = torch.where(x > max_val, max_val, x)
# RuntimeError: Expected condition, x and y to be on the same device, 
# but condition is on cuda:0 and x and y are on cpu and cuda:0 respectively
