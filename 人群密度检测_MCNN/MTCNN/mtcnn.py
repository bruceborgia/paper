import torch
import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1),   # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=2, stride=2),  # conv2
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),  # conv3
            nn.PReLU()
        )

        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_layer(x)

        cond = F.sigmoid(self.conv4_1(x))   # face classification
        box_offset = self.conv4_2(x)    # bounding bix regression
        land_offset = self.conv4_3(x)   # facial landmark localization

        return cond, box_offset, land_offset

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1),   # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # pool1
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()
        )
        self.line1 = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.PReLU()
        )

        self.line2_1 = nn.Linear(128, 1)
        self.line2_2 = nn.Linear(128, 4)
        self.line2_3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.view(x.size(0), -1)
        x = self.line1(x)

        label = F.sigmoid(self.conv5_1(x))
        box_offset = self.conv5_2(x)
        land_offset = self.conv5_3(x)

        return label, box_offset, land_offset

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  #conv2
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128),  # conv4
            nn.PReLU()
        )

        self.line1 = nn.Sequential(
            nn.Linear(128*3*3, 256),
            nn.PReLU()
        )

        self.line2_1 = nn.Linear(256, 1)
        self.line2_2 = nn.Linear(256, 4)
        self.line2_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)

        label = F.sigmoid(self.line2_1(x))
        box_offset = self.line2_2(x)
        land_offset = self.line2_3(x)

        return label, box_offset, land_offset



if __name__ == '__main__':
    print("hello")
