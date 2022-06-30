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
    def __init__(self, n_channels, mlp_depth, mlp_width, dtype, activ=nn.GELU, tau=1.6) -> None:
        super().__init__()
        self.tau = tau
        self.register_parameter('sigma', nn.Parameter(torch.rand(1)))

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
        '''
        type-inference network
        '''
        return self.mlp(x)
    
    def type_match(self, S, X):
        '''
        inputs:
            S:  (#funcs, #dtype)
            X:  (#tokens, #dtype)
        returns:
            C:  compatibility matrix of dim (#funcs, #tokens)
        '''
        T = self.forward(X) # (#tokens, #dtype)
        D = 1-torch.matmul(S, T.T) # (#funcs, #dtype)*(#dtype, #tokens) => (#funcs, #tokens)
        M = D>self.tau # mask signals less than tau => sparsity
        C_hat = torch.exp(-D/self.sigma)*M
        C = C_hat /(torch.sum(C_hat,dim=1,keepdim=True)+1e-4)
        return C

class ModLin(nn.Module):
    '''
    code:   embedding of dim dcond st. what function should do

    '''
    def __init__(self, code, dout, din, dcond, ) -> None:
        super().__init__()
        self.c = code
        self.register_parameter('w_c', nn.Parameter(torch.empty(din, dcond)))
        self.register_parameter('b', nn.Parameter(torch.empty(dout)))
        self.register_parameter('W', nn.Parameter(torch.empty(dout, din)))
        self.norm = torch.nn.LayerNorm(din)

    def forward(self, x):
        out = self.norm(torch.matmul(self.w_c, self.c))
        out = x*out
        out = torch.matmul(self.W,out)+self.b
        return out

class ModMLP(nn.Module):
    '''
    n_layers:   number of stacked ModLin blocks
    code:       code vector for each ModLin block => share same code
    '''
    def __init__(self, n_layers, code, dout, din, dcond, activ=nn.GELU) -> None:
        super().__init__()
        
        self.modlin_blocks = \
            [ModLin()]
    def forward(self, x):

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