import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .classifier import *

class EcgClassifer(nn.Module):
    def __init__(self, classifer=resnet34, num_classes=6, load_pretrain='ckpt2/resnet34_202311131734baseline/best_w.pth'):
        super().__init__()
        '''
        '''
        self.classifer = eval(classifer)(num_classes=num_classes)
        if load_pretrain:
            self.classifer.load_state_dict(torch.load(load_pretrain, map_location='cpu')['state_dict'])
            print(f"Classifier Load From {load_pretrain}")
        for param in self.classifer.parameters():
            param.requires_grad = False

    def forward(self, x):
        '''
        x: (b, lead, length)
        '''
        prob = self.classifer(x)
        return prob


