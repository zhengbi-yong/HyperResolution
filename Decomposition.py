import numpy as np
import tensorly as tl
import torch
from tensorly.decomposition import partial_tucker, tucker
from torch import nn

from SRCNN import SRCNN
from TVBMF import EVBMF


def tucker_rank(layer):
    W = layer.weight.data
    mode3 = tl.base.unfold(W, 0)
    mode4 = tl.base.unfold(W, 1)
    diag_0 = EVBMF(mode3)
    diag_1 = EVBMF(mode4)
    d1 = diag_0.shape[0]
    d2 = diag_1.shape[1]

    del mode3
    del mode4
    del diag_0
    del diag_1

    # round to multiples of 16
    return [int(np.ceil(d1 / 16) * 16) \
            , int(np.ceil(d2 / 16) * 16)]

def tucker_decomp(layer, rank):
    W = layer.weight.data

    core, [last, first] = partial_tucker(W, modes=[0,1], ranks=rank, init='svd')

    first_layer = nn.Conv2d(in_channels=first.shape[0],
                                       out_channels=first.shape[1],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)

    core_layer = nn.Conv2d(in_channels=core.shape[1],
                                       out_channels=core.shape[0],
                                       kernel_size=layer.kernel_size,
                                       stride=layer.stride,
                                       padding=layer.padding,
                                       dilation=layer.dilation,
                                       bias=False)

    last_layer = nn.Conv2d(in_channels=last.shape[1],
                                       out_channels=last.shape[0],
                                       kernel_size=1,
                                       padding=0,
                                       bias=True)
    
    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    fk = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    lk = last.unsqueeze_(-1).unsqueeze_(-1)

    first_layer.weight.data = fk
    last_layer.weight.data = lk
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return new_layers
    
def decompositionSRCNN(net, rank_func, decomp_func):
    i = 1
    while i < len(net.features):
        # find out the rank of the first conv layer
        layer_i = net.features[i]
        if not isinstance(layer_i, nn.Conv2d):
            i += 1
            continue
        
        layer_i = net.features[i]
        rank = rank_func(layer_i)
        print('rank of the {}th layer: {}'.format(i, rank))
        
        # debugging
        print("begin decomposing layer {}".format(i))
        decomp_layers = decomp_func(layer_i, rank)
        print("finished decomposing layer {}".format(i))

        net.features = nn.Sequential(\
        *(list(net.features[:i]) + decomp_layers + list(net.features[i + 1:])))

        i +=  len(decomp_layers)

    return net

def main():
    decomp_func = tucker_decomp
    decomp_arch = decompositionSRCNN
    rank_func = tucker_rank
    net = SRCNN() # 创建模型实例
    state_dict = torch.load("srcnn5.pth") # 加载.pth文件
    net.load_state_dict(state_dict) # 将参数赋值给模型
    net = decomp_arch(net, rank_func, decomp_func)
    torch.save({'arch':dict(net.named_children()),\
        'params': net.state_dict()},\
        'tucker_round_model.pth')

if __name__ == "__main__":
    main()