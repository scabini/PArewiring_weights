# Weight Organization Matters: Improving Deep Neural Network Random Initialization Through Neuronal Rewiring

## One of the things you need is a good neuronal organization!

![](https://github.com/scabini/network_science_weights/blob/main/rewiring_video.gif)



## Usage

- Given a pytorch model:

```python
from torch import nn
from weight_rewiring import PA_rewiring_np
for m in model.modules():
    #also works with any tensor with more than 1 dimension, just specify it here
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        PA_rewiring_np(m.weight)
```

