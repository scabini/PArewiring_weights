# network_science_weights
 
How to use:

- Given a pytorch model:

```python
from torch import nn
from weight_improvement import NS_weights
for m in model.modules():
    #also works with conv3d or any tensor with more than 1 dimension, just define it here
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        NS_weights(m.weight)
```

