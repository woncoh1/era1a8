import torch
import torch.nn as nn


def normlayer( # Normalization layer: batch, layer, or group
    n:str, # Normalization type
    o:int, # num_channels
    g:int=1, # num_groups
) -> nn.BatchNorm2d|nn.GroupNorm:
    NORMS = ['bn', 'gn', 'ln']
    assert n in NORMS, f"n should be one of {NORMS}"
    if n == 'bn': return nn.BatchNorm2d(o)
    if n == 'gn': return nn.GroupNorm(g, o)
    if n == 'ln': return nn.GroupNorm(1, o)


def convblock( # Convolution block: 3x3 convolution to extract features
    i:int, # in_channels
    o:int, # out_channels
    n:str, # Normalization type
    g:int, # num_groups
    d:float, # Dropout rate
    p:int=1, # padding
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, stride=1, padding=p, bias=False),
        normlayer(n, o, g=g),
        nn.Dropout2d(p=d),
        nn.ReLU(),
    )


def predblock( # Prediction block = GAP + softmax
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    return nn.Sequential(
        # [-1, i, s, s]
        nn.AdaptiveAvgPool2d(output_size=1),
        # [-1, i, 1, 1]
        nn.Conv2d(i, o, 1, stride=1, padding=0, bias=False),
        # [-1, o, 1, 1]
        nn.Flatten(),
        # [-1, o]
        nn.LogSoftmax(dim=1), # https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816
        # [-1, o]
    )


class SkipBlock(nn.Module):
    def __init__(self,
        i:int, # in_channels
        o:int, # out_channels
        n:str, # Normalization type
        g:int, # Normalization group count
        d:float, # Dropout rate
        down=False, # Downsample
    ) -> None:
        super().__init__()
        self.i = i
        self.o = o
        self.down = down
        self.conv1 = nn.Conv2d(i, o, 3, stride=2 if down else 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(o, o, 3, stride=1, padding=1, bias=False)
        self.norm1 = normlayer(n, o, g=g)
        self.norm2 = normlayer(n, o, g=g)
        self.drop1 = nn.Dropout2d(p=d)
        self.drop2 = nn.Dropout2d(p=d)
        if i != o:
            self.downsampler = nn.Conv2d(i, o, 1, stride=1, padding=0, bias=False)
        if down:
            self.downsampler = nn.Conv2d(i, o, 1, stride=2, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.drop1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.drop2(out)
        if (self.i != self.o) or self.down:
            identity = self.downsampler(identity)
        out = out + identity
        out = nn.functional.relu(out)
        return out


class Net(nn.Module):
    def __init__(self,
        norm:str='bn', # Normalization type
        grps:int=1, # Normalization group count
        drop:float=0, # Dropout rate
    ) -> None:
        super().__init__()
        self.conv1 = convblock( 3,  8, norm, grps, drop)
        self.conv2 = SkipBlock( 8,  8, norm, grps, drop)
        self.tran1 = SkipBlock( 8, 16, norm, grps, drop, down=True)
        self.conv3 = SkipBlock(16, 16, norm, grps, drop)
        self.conv4 = SkipBlock(16, 16, norm, grps, drop)
        self.conv5 = SkipBlock(16, 16, norm, grps, drop)
        self.tran2 = SkipBlock(16, 16, norm, grps, drop, down=True)
        self.conv6 = SkipBlock(16, 16, norm, grps, drop)
        self.conv7 = SkipBlock(16, 16, norm, grps, drop)
        self.conv8 = SkipBlock(16, 32, norm, grps, drop)
        self.tran3 = predblock(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tran1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.tran2(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.tran3(x)
        return x
