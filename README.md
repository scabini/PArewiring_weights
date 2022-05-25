# network_science_weights
 
How to use:

- Given a pytorch model:

```python
from torch import nn
for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        NS_weights(m.weight)
```

