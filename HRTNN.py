import os

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorly.decomposition import Tucker, tucker
from torch.nn import Module, Parameter, ParameterList


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
tl.set_backend('pytorch') # Or 'mxnet', 'numpy', 'tensorflow', 'cupy' or 'jax'
class HRTNN(nn.Module):
    def __init__(self):
        super(HRTNN, self).__init__() 
        self.decomp = Tucker(rank=[16,32,32], init='random')
        
        # y = w*x + b
        # initialize the weights and biases
        self.w1 = torch.rand(32,64,256).to('cuda')
        self.w2 = torch.rand(32,64,256).to('cuda')
        self.b = torch.rand(32,32,256,256).to('cuda')
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        # self.w1 = torch.rand(32,64,256)
        # self.w2 = torch.rand(32,64,256)
        # self.b = torch.rand(32,32,256,256)
        
        self.w1_tucker = self.decomp.fit_transform(self.w1)
        self.w2_tucker = self.decomp.fit_transform(self.w2)
        # print(type(self.w1_tucker[0]))
        # print(self.w1_tucker[0].size())
        # print(len(self.w1_tucker))
        # print(type(self.w1_tucker[1]))
        # print(len(self.w1_tucker[1]))
        self.w1_tucker[0] = Parameter(self.w1_tucker[0])
        for i in range(len(self.w1_tucker[1])):
            self.w1_tucker[1][i] = Parameter(self.w1_tucker[1][i])
        # self.w1_tucker = Parameter(self.w1_tucker)
        # self.w2_tucker = Parameter(self.w2_tucker)
        # self.b_tucker = self.decomp.fit_transform(self.b)
        self.c = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        # print(x.size())
        w1 = self.w1_tucker.to_tensor()
        w2 = self.w2_tucker.to_tensor()
        # print(type(self.w1_tucker))
        # print(self.w1_tucker)
        t1 = torch.einsum('abcd,bce->abde', x, w1)
        t1 = self.leakyrelu(t1)
        t2 = torch.einsum('abcd,bce->abde', t1, w2)
        # x = t2+self.b_tucker.to_tensor()
        t2 = self.leakyrelu(t2)
        x = t2+self.b
        x = self.c(x)
        return x