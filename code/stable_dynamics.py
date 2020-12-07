import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

#
# From StableDynamics.ipynb; Zico Kolter
#

# You can use this to compensate for numeric error:
VERIFY = False
V_SCALE = 0.01

global V_WRAP, SCALE_FX
V_WRAP = False
SCALE_FX = False
logger = logging.getLogger(__name__)


class ICNN(nn.Module):
    def __init__(self, layer_sizes, activation=F.relu_):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0])) 
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])
        self.act = activation
        self.reset_parameters()
        logger.info(f"Initialized ICNN with {self.act} activation")

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.act(z)

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)

        return F.linear(x, self.W[-1], self.bias[-1]) + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0]



class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)

class MakePSD(nn.Module):
    def __init__(self, f, n, eps=0.01, d=1.0):
        super().__init__()
        self.f = f
        self.zero = torch.nn.Parameter(f(torch.zeros(1,n)), requires_grad=False)
        self.eps = eps
        self.d = d
        self.rehu = ReHU(self.d)

    def forward(self, x):
        smoothed_output = self.rehu(self.f(x) - self.zero)
        quadratic_under = self.eps*(x**2).sum(1,keepdim=True)
        return smoothed_output + quadratic_under

class PosDefICNN(nn.Module):
    def __init__(self, layer_sizes, eps=0.1, negative_slope=0.05):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0])) 
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.eps = eps
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)

    def forward(self, x):
        z = F.linear(x, self.W[0])
        F.leaky_relu_(z, negative_slope=self.negative_slope)

        for W,U in zip(self.W[1:-1], self.U[:-1]):
            z = F.linear(x, W) + F.linear(z, F.softplus(U))*self.negative_slope
            z = F.leaky_relu_(z, negative_slope=self.negative_slope)

        z = F.linear(x, self.W[-1]) + F.linear(z, F.softplus(self.U[-1]))
        return F.relu(z) + self.eps*(x**2).sum(1)[:,None]




class Dynamics(nn.Module):
    def __init__(self, alpha=0.01,projfn="NN-REHU",h_dim=100,ph_dim=40,lsd=8):
        super().__init__()
        self.alpha = alpha
        self.ph_dim=ph_dim
        self.h_dim=h_dim

        fhat = nn.Sequential(nn.Linear(lsd, h_dim), nn.ReLU(),
                            nn.Linear(h_dim, h_dim), nn.ReLU(),
                            nn.Linear(h_dim, lsd))
        ## The convex function to project onto:
        projfn_eps =  0.01
        if projfn == "PSICNN":
            V = PosDefICNN([lsd, ph_dim, ph_dim, 1], eps=projfn_eps, negative_slope=0.3)
        elif projfn == "ICNN":
            V = ICNN([lsd, ph_dim, ph_dim, 1])
        elif projfn == "PSD":
            V = MakePSD(ICNN([lsd, ph_dim, ph_dim, 1]), lsd, eps=projfn_eps, d=1.0)
        elif projfn == "PSD-REHU":
            V = MakePSD(ICNN([lsd, ph_dim, ph_dim, 1], activation=ReHU(0.01)), lsd, eps=projfn_eps, d=1.0)
        elif projfn == "NN-REHU":
            seq = nn.Sequential(
                    nn.Linear(lsd, ph_dim,), nn.ReLU(),
                    nn.Linear(ph_dim, ph_dim), nn.ReLU(),
                    nn.Linear(ph_dim, 1), ReHU(0.01))
            V = MakePSD(seq, lsd, eps=projfn_eps, d=1.0)

        elif projfn == "EndPSICNN":
            V = nn.Sequential(nn.Linear(lsd, ph_dim, bias=False), nn.LeakyReLU(),
                nn.Linear(ph_dim, lsd, bias=False), nn.LeakyReLU(),
                PosDefICNN([lsd, ph_dim, ph_dim, 1], eps=projfn_eps, negative_slope=0.3))
        elif projfn == "NN":
            V = nn.Sequential(
                    nn.Linear(lsd, ph_dim,), nn.ReLU(),
                    nn.Linear(ph_dim, ph_dim), nn.ReLU(),
                    nn.Linear(ph_dim, 1))
                            
        self.fhat = fhat
        self.V = V


    def forward(self,x,g):
        x=torch.tensor(x,requires_grad=True)
        x0=abs(x-g)
        fx = self.fhat(x)
        if SCALE_FX:
            fx = fx / fx.norm(p=2, dim=1, keepdim=True).clamp(min=1.0)

        Vx = self.V(x) ##V(x)
        gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True, only_inputs=True)[0]## delta V(x)
        rv = fx - gV * (F.relu((gV*fx).sum(dim=1) + self.alpha*Vx[:,0])/(gV**2).sum(dim=1))[:,None]

        if VERIFY:
            # Verify that rv has no positive component along gV.
            # This helps us catch:
            #   (1) numeric error in the symbolic gradient calculation, and
            #   (2) Violation of the Lyapunov function when Euler integration is used.
            verify = (gV * rv).sum(dim=1)
            num_violation = len([v for v in verify if v > 0]) # (1)
            new_V = self.V(x + V_SCALE * rv)
            if (new_V > Vx).any(): # (2)
                err = sorted([v for v in (new_V - Vx).detach().cpu().numpy().ravel() if v > 0], reverse=True)
                logger.warn(f"V increased by: {err[:min(5, len(err))]} (total {len(err)}; upward grad {num_violation});")
        ##LF Risk
        X0=self.V(x0)
        # Lyapunov_risk = (F.relu(-Vx)+ 2*F.relu(gV+0.8)).mean()+1.2*(X0).pow(2)
        Lyapunov_risk = (F.relu(-Vx)+ 2*F.relu(gV)).mean()+10*(X0).pow(2).sum()
        # print(Vx.mean())
        # print(gV.mean())
        # print('risk',Lyapunov_risk)
        return Lyapunov_risk
