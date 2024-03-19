import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        bn=True,
        activation=nn.ReLU(),
        dropout=0,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels, output_channels, kernel_size=kernel_size, stride=stride, bias=not bn
            ),
            activation,
            nn.BatchNorm2d(output_channels) if bn else nn.Identity(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)