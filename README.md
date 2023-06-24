# TSAI ERA V1 A8: Normalization in CNN with CIFAR
> Training a fully convolutional neural network model on the CIFAR-10 dataset using various normalizations
- CIFAR-10: Canadian Institute For Advanced Research dataset with 10 classes ([Papers With Code](https://paperswithcode.com/dataset/cifar-10))
- CNN: Convolutional Neural Network model ([Stanford cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks))

## Installation
If you want to use our models, dataloaders, training engine, and other utilities, please run the following command:
```console
git clone https://github.com/woncoh1/era1a8.git
```
And then import the modules in Python:
```python
from era1a7.src import data, models, engine, utils
```

## Objectives
Use batch, layer, and group normalization to achieve all the followings:
- Test accuracy > 70.0 %
- Number of parameters < 50,000
- Number of epochs <= 20

## Experiments
1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a8/blob/main/nbs/S8_BN.ipynb) Batch normalization
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a8/blob/main/nbs/S8_LN.ipynb) Layer normalization
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a8/blob/main/nbs/S8_GN.ipynb) Group normalization

## Results summary
- Test accuracy
    - Batch: ? % 
    - Layer: ? %
    - Group: ? %
- Number of Parameters: 48,712
- Number of Epochs: 20

## References
- https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
- https://github.com/parrotletml/era_session_seven
