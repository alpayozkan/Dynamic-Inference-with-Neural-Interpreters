from turtle import forward
import torch
import torch.nn as nn


class TypeInference(nn.Module):
    '''
    Type Inference module
    Given input x => linear projections with mlp => obtain type embedding for that input
    with which we will match most compatible functions

    n_channels: input channel dim
    mlp_depth:  number of layers
    mlp_width:  number of neurons in a layer
    dtype:      output dimensionality
    acitv:      type of activation function in mlp
    '''
    def __init__(self, n_channels, mlp_depth, mlp_width, dtype, activ=nn.GELU) -> None:
        super().__init__()
        net = \
        [
            [
                nn.Linear(n_channels, mlp_width),
                activ(),
            ]
        ] + \
        [
            [
                nn.Linear(mlp_width, mlp_width),
                activ(),
            ]   for i in range(mlp_depth-1)
        ] + \
        [
            [
                nn.Linear(mlp_width, dtype),
            ]
        ]

        net = sum(net, [])
        self.mlp = nn.Sequential(*net)

    def forward(self, x):
        return self.mlp(x)

class ModLin(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return None

class FuncBlock(nn.Module):
    '''
    s:  function signature of dimension (dtype, )
    '''
    def __init__(self, dtype) -> None:
        super().__init__()
        self.register_parameter('s', nn.Parameter(torch.empty(dtype)))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.s)
    def forward(self, x):
        return None