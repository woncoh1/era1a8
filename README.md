# Normalization in CNN with CIFAR
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
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a8/blob/main/nbs/S8_GN.ipynb) Group normalization
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/woncoh1/era1a8/blob/main/nbs/S8_LN.ipynb) Layer normalization

## Results summary
- Test accuracy
    - Batch: 81.67 %
    - Group: 74.97 %
    - Layer: 73.70 %
- Number of Parameters: 48,136
- Number of Epochs: 20

## Model summary
```
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
Net (Net)                                [128, 10]                 --
├─Sequential (conv1)                     [128, 8, 32, 32]          --
│    └─Conv2d (0)                        [128, 8, 32, 32]          216
│    └─GroupNorm (1)                     [128, 8, 32, 32]          16
│    └─Dropout2d (2)                     [128, 8, 32, 32]          --
│    └─ReLU (3)                          [128, 8, 32, 32]          --
├─SkipBlock (conv2)                      [128, 8, 32, 32]          --
│    └─Conv2d (conv1)                    [128, 8, 32, 32]          576
│    └─GroupNorm (norm1)                 [128, 8, 32, 32]          16
│    └─Dropout2d (drop1)                 [128, 8, 32, 32]          --
│    └─Conv2d (conv2)                    [128, 8, 32, 32]          576
│    └─GroupNorm (norm2)                 [128, 8, 32, 32]          16
│    └─Dropout2d (drop2)                 [128, 8, 32, 32]          --
├─SkipBlock (tran1)                      [128, 16, 16, 16]         --
│    └─Conv2d (conv1)                    [128, 16, 16, 16]         1,152
│    └─GroupNorm (norm1)                 [128, 16, 16, 16]         32
│    └─Dropout2d (drop1)                 [128, 16, 16, 16]         --
│    └─Conv2d (conv2)                    [128, 16, 16, 16]         2,304
│    └─GroupNorm (norm2)                 [128, 16, 16, 16]         32
│    └─Dropout2d (drop2)                 [128, 16, 16, 16]         --
│    └─Conv2d (downsampler)              [128, 16, 16, 16]         128
├─SkipBlock (conv3)                      [128, 16, 16, 16]         --
│    └─Conv2d (conv1)                    [128, 16, 16, 16]         2,304
│    └─GroupNorm (norm1)                 [128, 16, 16, 16]         32
│    └─Dropout2d (drop1)                 [128, 16, 16, 16]         --
│    └─Conv2d (conv2)                    [128, 16, 16, 16]         2,304
│    └─GroupNorm (norm2)                 [128, 16, 16, 16]         32
│    └─Dropout2d (drop2)                 [128, 16, 16, 16]         --
├─SkipBlock (conv4)                      [128, 16, 16, 16]         --
│    └─Conv2d (conv1)                    [128, 16, 16, 16]         2,304
│    └─GroupNorm (norm1)                 [128, 16, 16, 16]         32
│    └─Dropout2d (drop1)                 [128, 16, 16, 16]         --
│    └─Conv2d (conv2)                    [128, 16, 16, 16]         2,304
│    └─GroupNorm (norm2)                 [128, 16, 16, 16]         32
│    └─Dropout2d (drop2)                 [128, 16, 16, 16]         --
├─SkipBlock (conv5)                      [128, 16, 16, 16]         --
│    └─Conv2d (conv1)                    [128, 16, 16, 16]         2,304
│    └─GroupNorm (norm1)                 [128, 16, 16, 16]         32
│    └─Dropout2d (drop1)                 [128, 16, 16, 16]         --
│    └─Conv2d (conv2)                    [128, 16, 16, 16]         2,304
│    └─GroupNorm (norm2)                 [128, 16, 16, 16]         32
│    └─Dropout2d (drop2)                 [128, 16, 16, 16]         --
├─SkipBlock (tran2)                      [128, 16, 8, 8]           --
│    └─Conv2d (conv1)                    [128, 16, 8, 8]           2,304
│    └─GroupNorm (norm1)                 [128, 16, 8, 8]           32
│    └─Dropout2d (drop1)                 [128, 16, 8, 8]           --
│    └─Conv2d (conv2)                    [128, 16, 8, 8]           2,304
│    └─GroupNorm (norm2)                 [128, 16, 8, 8]           32
│    └─Dropout2d (drop2)                 [128, 16, 8, 8]           --
│    └─Conv2d (downsampler)              [128, 16, 8, 8]           256
├─SkipBlock (conv6)                      [128, 16, 8, 8]           --
│    └─Conv2d (conv1)                    [128, 16, 8, 8]           2,304
│    └─GroupNorm (norm1)                 [128, 16, 8, 8]           32
│    └─Dropout2d (drop1)                 [128, 16, 8, 8]           --
│    └─Conv2d (conv2)                    [128, 16, 8, 8]           2,304
│    └─GroupNorm (norm2)                 [128, 16, 8, 8]           32
│    └─Dropout2d (drop2)                 [128, 16, 8, 8]           --
├─SkipBlock (conv7)                      [128, 16, 8, 8]           --
│    └─Conv2d (conv1)                    [128, 16, 8, 8]           2,304
│    └─GroupNorm (norm1)                 [128, 16, 8, 8]           32
│    └─Dropout2d (drop1)                 [128, 16, 8, 8]           --
│    └─Conv2d (conv2)                    [128, 16, 8, 8]           2,304
│    └─GroupNorm (norm2)                 [128, 16, 8, 8]           32
│    └─Dropout2d (drop2)                 [128, 16, 8, 8]           --
├─SkipBlock (conv8)                      [128, 32, 8, 8]           --
│    └─Conv2d (conv1)                    [128, 32, 8, 8]           4,608
│    └─GroupNorm (norm1)                 [128, 32, 8, 8]           64
│    └─Dropout2d (drop1)                 [128, 32, 8, 8]           --
│    └─Conv2d (conv2)                    [128, 32, 8, 8]           9,216
│    └─GroupNorm (norm2)                 [128, 32, 8, 8]           64
│    └─Dropout2d (drop2)                 [128, 32, 8, 8]           --
│    └─Conv2d (downsampler)              [128, 32, 8, 8]           512
├─Sequential (tran3)                     [128, 10]                 --
│    └─AdaptiveAvgPool2d (0)             [128, 32, 1, 1]           --
│    └─Conv2d (1)                        [128, 10, 1, 1]           320
│    └─Flatten (2)                       [128, 10]                 --
│    └─LogSoftmax (3)                    [128, 10]                 --
==========================================================================================
Total params: 48,136
Trainable params: 48,136
Non-trainable params: 0
Total mult-adds (M): 982.64
==========================================================================================
Input size (MB): 1.57
Forward/backward pass size (MB): 145.76
Params size (MB): 0.19
Estimated Total Size (MB): 147.53
==========================================================================================
```

## References
- https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
- https://github.com/parrotletml/era_session_seven
