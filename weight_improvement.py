# -*- coding: utf-8 -*-
"""
weight optimizer

@author: scabini
"""

import torch
from numpy.random import default_rng

#manual parameters to make the stochastic operation reproducible or not
reproducible = True 
base_seed = 666999

def PA_rewiring(weights_out):
    with torch.no_grad():
        dimensions = weights_out.shape 
        for neuron in range(1,dimensions[0]):
            strength = torch.sum(weights_out[0:neuron], axis=0)        
            strength = strength + torch.abs(torch.min(strength)) + 1
            probs = strength / torch.sum(strength)
            probs = probs.cpu().detach().numpy()
            if reproducible:
                rng = default_rng(base_seed*neuron)
            else:
                rng = default_rng()            
            selected_neurons = rng.choice(a=[i for i in range(dimensions[1])], size=dimensions[1],p=probs, replace=False)   
            edges_to_rewire = torch.argsort(weights_out[neuron])
            weights_out[neuron, selected_neurons] = weights_out[neuron, edges_to_rewire]                
    return weights_out


def NS_weights(weights): 
    mode='fan_in_out'
    if weights.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    with torch.no_grad():  
        output_neurons = weights.size(0)
        input_neurons = weights.numel() // output_neurons
    
        dimensions = weights.shape 
            
        weights = weights.reshape((output_neurons, input_neurons))
        if mode =='fan_in' or mode =='fan_in_out':            
            PA_rewiring(weights)   
            
        if mode =='fan_out' or mode =='fan_in_out':
            PA_rewiring(torch.transpose(weights, 0, 1))
        
        weights = weights.reshape((dimensions))      
      
    return weights


### HOW TO USE:
# eg. initializing all layers of a model:
    
# from torch import nn
# for m in model.modules():
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         NS_weights(m.weight)
    
    
