# Improving Deep Neural Network Random Initialization Through Neuronal Rewiring

https://arxiv.org/abs/2207.08148

## Weight organization matters! One of the things you need is a good neuronal organization.

![](https://github.com/scabini/network_science_weights/blob/main/rewiring_video.gif)

We propose the Preferential Attachment (PA) Rewiring technique for minimizing the strength of randomly initialized neurons. The reorganized weights improves training and generalization while also reducing performance variance. See some results below for training 100 models on CIFAR-10, for each architecture, by varying only the seed used for random weight sampling (while all other seeds and stochastic processes, such as data augmentation, are fixed).

<p align="center">
    <img src="some_results.jpg">
</p>


## Requirements

Only pytorch and numpy is needed (see _torch and _np versions of the method in "weight_rewiring.py")

## Usage

- Given a pytorch model, the preferential attachment rewiring of the weights at each layer is achieved with:

```python
import torch
import weight_rewiring
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, verbose =False)
for m in model.modules():
    #also works with any tensor with more than 1 dimension, just specify it here
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear): 
        torch.nn.init.orthogonal_(m.weight) #select a base weight distribution, or ignore this line to keep pytorch's standard init
        weight_rewiring.PA_rewiring_np(m.weight)
```

- You can also perform the random minimization of the strength variance (see Table 1 and Figure 4 in the paper)

```python
        weight_rewiring.stabilize_strength(torch.nn.init.orthogonal_, m.weight)
```

- Or even both operations (in this order!):

```python
        weight_rewiring.stabilize_strength(torch.nn.init.orthogonal_, m.weight)
        weight_rewiring.PA_rewiring_np(m.weight)
```

## Reference

If you use our code or methods, please cite:

Scabini, Leonardo, Bernard De Baets, and Odemir M. Bruno. "Improving Deep Neural Network Random Initialization Through Neuronal Rewiring." arXiv preprint arXiv:2207.08148 (2022).

```
@article{scabini2022improving,
  title={Improving Deep Neural Network Random Initialization Through Neuronal Rewiring},
  author={Scabini, Leonardo and De Baets, Bernard and Bruno, Odemir M},
  journal={arXiv preprint arXiv:2207.08148},
  year={2022}
}
```   
