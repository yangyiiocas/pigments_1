import torch
import torch.nn as nn

import numpy as np
import random
import pickle



class Group(nn.Module):
    def __init__(self, in_channels,inner_channels,out_channels,above_channels):
        super(Group, self).__init__()

        self.fn1 =nn.Linear(in_features=in_channels, out_features=inner_channels, bias=True)
        self.fn2 =nn.Linear(in_features=inner_channels, out_features=inner_channels, bias=True)
#         self.fn_p =nn.Linear(in_features=inner_channels, out_features=inner_channels, bias=True)
        self.fn3 =nn.Linear(in_features=inner_channels+above_channels, out_features=out_channels, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x, x_above):
        x = self.relu( self.fn1(x) )
        x = self.relu( self.fn2(x) )
        if x_above is not None:
            x = torch.cat([x,x_above],axis=1)
        x = self.fn3(x)
        return x

class Block(nn.Module):
    def __init__(self,in_channels,inner_channels,out_channels=1):
        super(Block, self).__init__()
        self.group0 = Group(in_channels=in_channels,inner_channels=8,out_channels=8,above_channels=0)
        self.group1 = Group(in_channels=in_channels,inner_channels=16,out_channels=16,above_channels=8)
        self.group2 = Group(in_channels=in_channels,inner_channels=32,out_channels=32,above_channels=16)
        self.group3 = Group(in_channels=in_channels,inner_channels=inner_channels,out_channels=out_channels,above_channels=32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.group0(x,None)
        x1 = self.group1(x,self.relu(x0))
        x2 = self.group2(x,self.relu(x1))
        x3 = self.group3(x,self.relu(x2))
        return x3


class NN(nn.Module):
    def __init__(self,in_channels,inner_channels):
        super(NN, self).__init__()

        self.Block1 = Block(in_channels,inner_channels,out_channels=inner_channels)
        self.Block2 = Block(inner_channels,inner_channels,out_channels=inner_channels)
        self.Block3 = Block(inner_channels,inner_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1):
        y1 = self.Block1(x1)
        x2 = self.relu(y1)
        y2 = self.Block2(x2)
        x3 = self.relu(y2)
        y3 = self.Block3(x2+x3)
        return y3 
