import torch.nn as nn
from nn.conv_block import ConvBlock


class CNN(nn.Module):
    def __init__(
        self,
        input_channels=3,  # how many channels do inputs have?
        conv_channels=[
            8,
            16,
            16,
            32,
            32,
            64,
            64,
            128,
            128
        ],  # how many convolutional layers (and how many channels)?
        strides=None,  # specify conv. layers' strides (None => automatically set to 1)
        kernel_sizes=[
            11,
            9,
            7,
            7,
            5,
            5,
            3,
            3,
            3,
        ],  # kernel sizes for each conv. layer (None => automatically set to 3)
        batch_norm=False,  # whether or not to use BatchNorm
        activation=nn.LeakyReLU(),  # what non-linearity to use?
        mlp_layers=[
            128,
            64,
            32,
            2,
        ],  # how many (and how wide) fully-connected layers in the final predictor
        dropout=0,  # probability of setting "neurons" to 0 at train-time
    ):
        super().__init__()

        # initialize kernel sizes and strides if not provided
        if kernel_sizes is None:
            kernel_sizes = [3] * len(conv_channels)
        if strides is None:
            strides = [1] * len(conv_channels)

        # convolutional layers' number of channels
        channels = [input_channels] + conv_channels

        conv_layers = []

        for i in range(len(channels) - 1):
            input_c = channels[i]
            output_c = channels[i + 1]
            # create a convolutional layer
            conv_layers.append(
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_sizes[i],
                    strides[i],
                    batch_norm,
                    activation,
                    dropout,
                )
            )

        # turn the list into a torch.nn.Module that automatically
        # chains .forward() calls
        self.conv_layers = nn.Sequential(*conv_layers)

        # average features across the spatial dimension.
        # the output shape will be batch_size x conv_channels[-1] x 1
        self.aggregation = nn.AdaptiveAvgPool1d(1)

        mlp = []
        mlp_sizes = [conv_channels[-1]] + mlp_layers
        for i in range(len(mlp_sizes) - 1):
            # create a linear layer
            mlp.append(nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]))
            # add non-linearity
            mlp.append(activation)
            # add dropout
            mlp.append(nn.Dropout(dropout))

        # again, turn list into module
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # how to process a sample through the model?

        # compute the output feature maps after all convolutional layers
        y = self.conv_layers(x)
        # aggregate features into a vector
        y = self.aggregation(y)
        # reshape to batch_size x conv_channels[-1] for the mlp
        y = y.view(y.size(0), -1)
        # forward through the mlp
        y = self.mlp(y)
        return y