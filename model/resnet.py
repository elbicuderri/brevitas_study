import torch
import torch.nn as nn

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.block11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
        )

        self.block12 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
        )

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False)

        self.block21 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
        )

        self.block22 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False)

        self.block31 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.block32 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        batch = x.size(0)
        out0 = self.conv0(x)

        res11 = out0
        out11 = self.block11(out0)
        out11 += res11
        out11 = self.relu(out11)

        res12 = out11
        out12 = self.block12(out11)
        out12 += res12
        out12 = self.relu(out12)

        res21 = self.conv2(out12)
        out21 = self.block21(out12)
        out21 += res21
        out21 = self.relu(out21)

        res22 = out21
        out22 = self.block22(out21)
        out22 += res22
        out22 = self.relu(out22)

        res31 = self.conv3(out22)
        out31 = self.block31(out22)
        out31 += res31
        out31 = self.relu(out31)

        res32 = out31
        out32 = self.block32(out31)
        out32 += res32
        out32 = self.relu(out32)

        out4 = self.avg_pool(out32)
        out4 = out4.view(batch, -1)
        out = self.fc(out4)

        return out