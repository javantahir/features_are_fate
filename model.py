import numpy as np
import torch
from torch import nn

class deepLinear(nn.Module):
    def __init__(self, L, d, instances, alpha, init='rand'):
        super().__init__()
        self.L = L
        self.d = d
        self.instances = instances
        self.alpha = alpha
        if init == 'zas':
            self.layers = nn.ParameterList([nn.Parameter(torch.tile(alpha*torch.eye(d).unsqueeze(0), (instances,1,1))) for k in range(L-1)] \
                                        + [nn.Parameter(torch.tile(alpha*torch.ones(1,d,1), (instances,1,1)))])
        elif init == 'rand':
            self.layers = nn.ParameterList([nn.Parameter(torch.tile(alpha*torch.randn(1,d,d)/np.sqrt(d), (instances,1,1))) for k in range(L-1)] \
                                        + [nn.Parameter(torch.tile(alpha*torch.randn(1,d,1), (instances,1,1)))])
            
    def duplicate(self, n):
        if self.instances > 1:
            raise ValueError("cannot duplicate a model with instances > 1")
        self.instances = n
        layers = []
        for layer in self.layers:
            layers.append(nn.Parameter(torch.tile(layer.clone(), (n,1,1))))
        self.layers = nn.ParameterList(layers)

    def getHiddenFeatures(self):
        feat = self.layers[0]
        for l in range(len(self.layers) - 2):
            feat = feat @ self.layers[l+1]
        
        return feat
    
    def getBeta(self):
        beta_hat = self.layers[0]
        for l in range(len(self.layers)-1):
            beta_hat = beta_hat @ self.layers[l+1] 
            
        return beta_hat
    
    def forward(self, x):
        # x has shape instances, n, d
        out = x
        for layer in self.layers: 
            out = torch.einsum("...jk, ...kl -> ...jl",out,layer)
            
        return out.squeeze(-1)
    
'''
import train
L = 5
instances = 5
d = 1000
alpha = 1
sigma = 0.01
model = deepLinear(L, d, instances, alpha, init='rand')

beta = torch.randn((d,1))
Beta = torch.tile(beta.unsqueeze(0), (instances,1,1))
X, y, dataset, dataloader = train.makeGaussianLinearDataset((instances, 200, d), Beta, noise=sigma)
print(model(X).shape)

for l in range(len(model.layers) - 1):
    Wl = model.layers[l]
    Wl1 = model.layers[l+1]
    mat = Wl.transpose(-2,-1) @ Wl - Wl1 @ Wl1.transpose(-2,-1)
    print(torch.all(torch.linalg.svdvals(mat) > 0))
'''