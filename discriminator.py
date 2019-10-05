import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(
        self,
        channel_counts = [64, 128, 256],
        kernel_size = 11,
        block_size = 3,
        activation = lambda x: nn.functional.leaky_relu(x, negative_slope = 0.3),
        fc_layers = 2,
        fc_nodes = 1024,
        dropout = 0.2,
        training = True,
    ):
        super(Discriminator, self).__init__()

        # Store hyperparameters
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.dropout = dropout
        self.activation = activation

        # Initialize all layer containers
        self.downsample_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        in_channels = 1
        for out_channels in channel_counts:
            for i in range(block_size):
                self.downsample_layers.append(self.conv_layer(in_channels, out_channels, downsample = in_channels != 1))
                in_channels = out_channels

        for layer in range(fc_layers):
            self.fc_layers.append(nn.Linear(in_channels, fc_nodes))
            in_channels = fc_nodes

        self.fc_layers.append(nn.Linear(in_channels, 1))


    # Define a single convolutional layer
    def conv_layer(self, in_channels, out_channels, downsample = False):

        return nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = self.kernel_size,
            stride = 2 if downsample else 1,
            padding = self.kernel_size // 2,
        )

    # Define the forward pass of our model, unet-style
    def forward(self, x):

        # Apply downsampling layers
        for i, layer in enumerate(self.downsample_layers):
            x = self.activation(layer(x))

            # dropout once each block
            if i % self.block_size == 0:
                x = nn.functional.dropout(x, p = self.dropout)

        x = x.transpose(1, 2)

        for layer in self.fc_layers:
            x = self.activation(layer(x))
            x = nn.functional.dropout(x, p = self.dropout)

        # Reduce to 1 channel and scale from -1 to 1
        return torch.sigmoid(x).mean()
