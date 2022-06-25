# Weight Organization Matters: Improving Deep Neural Network Random Initialization Through Neuronal Rewiring

## One of the things you need is a good neuronal organization!

![](https://github.com/scabini/network_science_weights/blob/main/rewiring_video.gif)



## Usage

- Given a pytorch model:

```python
import torch
import weight_rewiring
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, verbose =False)
for m in model.modules():
    #also works with any tensor with more than 1 dimension, just specify it here
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear): 
        torch.nn.init.orthogonal_(m.weight) #select a base weight distribution
        weight_rewiring.PA_rewiring_np(m.weight)
```

