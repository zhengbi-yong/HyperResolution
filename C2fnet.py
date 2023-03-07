from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, -0.0001, 0.01)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)


class BottleneckC2f(nn.Module):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, kernels, groups, expand
    def __init__(self, c1, c2, g=1, k=[3, 3], e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, k[0], 1, k[0] // 2),
            nn.InstanceNorm2d(c_)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c2, k[1], 1, k[1] // 2, groups=g),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1),
            nn.InstanceNorm2d(2 * self.c),
            nn.PReLU(2 * self.c, init=0.2)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1),
            nn.InstanceNorm2d(c2),
            nn.PReLU(c2, init=0.2)
        )  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BottleneckC2f(
            self.c, self.c, g, k=[3, 3], e=0.5) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fnet(nn.Module):
    def __init__(self, in_channels=32, out_channels=1):
        super(C2fnet, self).__init__()

        self.feblock1 = C2f(in_channels, 64)
        self.feblock2 = C2f(64, 128)
        self.feblock3 = C2f(128, 256)
        self.feblock4 = C2f(256, 64)

    def forward(self, x):
        f1 = self.feblock1(x)
        f2 = self.feblock2(f1)
        f3 = self.feblock3(f2)
        f4 = self.feblock4(f3)
        return f4


if __name__ == "__main__":
    inputdata = torch.ones((2, 32, 64, 64))
    model = C2fnet()
    output = model(inputdata)
    print(output.shape)
