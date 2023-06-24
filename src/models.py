import torch
import torch.nn as nn


def norm( # Normalization layer: batch, layer, or group
    n:str, # Normalization type
    o:int, # num_channels
    g:int, # num_groups
) -> nn.modules:
    NORMS = ['bn', 'ln', 'gn',]
    assert n in NORMS, f"n should be one of {NORMS}"
    if n == 'bn': return nn.BatchNorm2d(o)
    if n == 'ln': return nn.GroupNorm(1, o)
    if n == 'gn': return nn.GroupNorm(g, o)


def conv( # Convolution layer: 3x3 convolution to extract features
    i:int, # in_channels
    o:int, # out_channels
    n:str, # Normalization type
    g:int, # num_groups
    d:float, # Dropout rate
    p:int=1, # padding
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, stride=1, padding=p, padding_mode='replicate'),
        norm(n, o, g=g),
        nn.Dropout2d(p=d),
        nn.ReLU(),
    )


def last( # Prediction layer = GAP + softmax
    i:int, # in_channels
    o:int, # out_channels
) -> nn.Sequential:
    return nn.Sequential(
        # [-1, i, s, s]
        nn.AdaptiveAvgPool2d(output_size=1),
        # [-1, i, 1, 1]
        nn.Conv2d(i, o, 1, stride=1),
        # [-1, o, 1, 1]
        nn.Flatten(),
        # [-1, o]
        nn.LogSoftmax(dim=1), # https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816
        # [-1, o]
    )


class SkipBlock(nn.Module):
    # https://engineering.purdue.edu/DeepLearn/pdf-kak/SkipConsAndBN.pdf
    # https://engineering.purdue.edu/kak/distDLS/DLStudio-2.3.0_CodeOnly.html
    def __init__(self,
        i:int, # in_channels
        o:int, # out_channels
        n:str, # Normalization type
        g:int, # num_groups
        d:float, # Dropout rate
        down=False, # Downsampling
        skip=True, # Skip connection
    ) -> None:
        super().__init__()
        self.down = down
        self.skip = skip
        self.i = i
        self.o = o
        self.conv1 = nn.Conv2d(i, o, 3, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(i, o, 3, stride=1, padding=1, padding_mode='replicate')
        self.norm1 = norm(n, o, g=g)
        self.norm2 = norm(n, o, g=g)
        self.drop1 = nn.Dropout2d(p=d)
        self.drop2 = nn.Dropout2d(p=d)
        if down:
            self.downsampler = nn.Conv2d(i, o, 1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.drop1(out)
        out = nn.functional.relu(out)
        if self.i == self.o:
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.drop2(out)
        if self.down:
            out = self.downsampler(out)
            identity = self.downsampler(identity)
        if self.skip:
            if self.i == self.o:
                out = out + identity
            else:
                out = torch.cat([
                    #  [B,C      ,H,W]
                    out[:,:self.i,:,:] + identity,
                    out[:,self.i:,:,:] + identity,
                ], dim=1)
        out = nn.functional.relu(out)
        return out


class Net(nn.Module):
    def __init__(self,
        norm:str='bn',
        grps:int=1,
        drop:float=0,
    ) -> None:
        super().__init__()
        self.conv1 = conv(3, 8, norm, grps, drop)
        self.conv2 = SkipBlock(8, 8, norm, grps, drop)
        self.tran1 = SkipBlock(8, 8, norm, grps, drop, down=True)
        self.conv3 = SkipBlock(8, 8, norm, grps, drop)
        self.conv4 = SkipBlock(8, 16, norm, grps, drop)
        self.conv5 = SkipBlock(16, 16, norm, grps, drop)
        self.tran2 = SkipBlock(16, 16, norm, grps, drop, down=True)
        self.conv6 = SkipBlock(16, 16, norm, grps, drop)
        self.conv7 = SkipBlock(16, 32, norm, grps, drop)
        self.conv8 = SkipBlock(32, 32, norm, grps, drop)
        self.tran3 = last(32, 10)

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
