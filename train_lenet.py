import torch
from torch.nn import Module
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear
from brevitas.core.quant import QuantType
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from statistics import mean
import numpy as np
from torchsummary import summary
from quan_model.lenet_quan import QuantLeNet

class QuantLeNet(Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.quant_inp = QuantIdentity(bit_width=8)
        self.conv1 = QuantConv2d(3, 6, 5, weight_bit_width=4)
        self.relu1 = QuantReLU(bit_width=8)
        self.conv2 = QuantConv2d(6, 16, 5, weight_bit_width=4)
        self.relu2 = QuantReLU(bit_width=8)
        self.fc1   = QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4)
        self.relu3 = QuantReLU(bit_width=8)
        self.fc2   = QuantLinear(120, 84, bias=True, weight_bit_width=4)
        self.relu4 = QuantReLU(bit_width=8)
        self.fc3   = QuantLinear(84, 10, bias=False, weight_bit_width=4)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = QuantLeNet().to(device=device)

batch_size = 64
epochs = 50

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
 transforms.ToTensor(), # 0 ~ 1
 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
]) # output[channel] = (input[channel] - mean[channel]) / std[channel]

train_dataset = datasets.CIFAR10('~/data/cifar10/train/',
                                 train=True,
                                 download=True,
                                 transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)

valid_dataset = datasets.CIFAR10(root='~/data/cifar10/test/',
                                            train=False, 
                                            download=True,
                                            transform=transform)

valid_loader = DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

loss_dict = {}
val_loss_dict = {}
acc_dict = {}
val_acc_dict = {}
train_step = len(train_loader)
val_step = len(valid_loader)

for epoch in range(1, epochs + 1):
    loss_list = [] # losses of i'th epoch
    num_correct = 0
    num_samples = 0
    for train_step_idx, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        
        model.train()
        output = model(img)
        loss = loss_fn(output, label)
        _, predictions = output.max(1)
        num_correct += (predictions == label).sum()
        num_samples += predictions.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if ((train_step_idx+1) % 100 == 0):
            print(f"Epoch [{epoch}/{epochs}] Step [{train_step_idx + 1}/{train_step}] Loss: {loss.item():.4f} Accuracy: {(num_correct / num_samples) * 100:.4f}")

    loss_dict[epoch] = loss_list
    acc_dict[epoch] = (num_correct / num_samples) * 100

    val_loss_list = []
    val_num_correct = 0
    val_num_samples = 0
    for val_step_idx, (val_img, val_label) in enumerate(valid_loader):
        with torch.no_grad():
            val_img = val_img.to(device)
            val_label = val_label.to(device)
            
            model.eval()
            val_output = model(val_img)
            val_loss = loss_fn(val_output, val_label)
            _, val_predictions = val_output.max(1)
            val_num_correct += (val_predictions == val_label).sum()
            val_num_samples += val_predictions.size(0)

        val_loss_list.append(val_loss.item())

    val_loss_dict[epoch] = val_loss_list
    val_acc_dict[epoch] = (val_num_correct / val_num_samples) * 100

    torch.save(
        {
        f"epoch": epoch,
        f"model_state_dict": model.state_dict(),
        f"optimizer_state_dict": optimizer.state_dict(),
        f"loss": mean(loss_dict[epoch]),
        f"accuracy": acc_dict[epoch]
        },
        f"checkpoint/quantlenet_epoch_{epoch}.ckpt")

    print(f"Epoch [{epoch}] Train Loss: {mean(loss_dict[epoch]):.4f} Val Loss: {mean(val_loss_dict[epoch]):.4f}")
    print(f"Epoch [{epoch}] Train Accuracy: {acc_dict[epoch]:.4f} Val Accuracy: {val_acc_dict[epoch]:.4f}")
    print("========================================================================================")

torch.save(model.state_dict(), 'weight/quantlenet.pt')
