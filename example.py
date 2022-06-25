# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:09:04 2022

Sample code for using PA rewiring

@author: scabini
"""

import torch
import weight_rewiring

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, verbose =False)

for m in model.modules():
    #define the types of layers to rewire (models usually contains only Conv and Linear layers)
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear): 
        
        # weight_rewiring.PA_rewiring_torch(m.weight) #torch version has lower precision on the strength calculation
        
        print('max(|si|) before rewiring:', torch.max(torch.abs(torch.sum(m.weight, dim=0))))
        
        weight_rewiring.PA_rewiring_np(m.weight) #np version was used in the paper
        
        print('           after rewiring:', torch.max(torch.abs(torch.sum(m.weight, dim=0))))
        print('')

