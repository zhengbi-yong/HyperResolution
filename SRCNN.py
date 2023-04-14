import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # # patch extraction layer
        # self.conv1 = nn.Conv2d(32, 64, kernel_size=9, padding=4)
        # # non-linear mapping layer
        # self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        # # reconstruction layer
        # self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.features = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        # upsample the input image by a factor of 4 using bicubic interpolation
        x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        # apply the three convolutional layers with ReLU activation
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.conv3(x)
        x = self.features(x)
        return x