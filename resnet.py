import torch
import torch.nn as nn

class ResNet(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim, 
        channel_counts = [64, 128, 256, 512],
        dense_count = 2,
        dense_nodes = 1024,
        activation = lambda x: nn.functional.leaky_relu(x, negative_slope = 0.3),
        dropout = 0.3,
        training = True,
    ):
        super(ResNet, self).__init__()

        # Store hyperparameters
        self.activation = activation
        self.dropout = dropout
        self.training = training
        self.output_dim = output_dim

        # Define conv layers
        in_channels = 1
        self.conv_layers = nn.ModuleList()
        for out_channels in channel_counts:
            self.conv_layers.extend(conv_block(in_channels, out_channels))
            in_channels = out_channels

        # Define dense layers
        self.dense_layers = nn.ModuleList()
        self.in_features = out_channels * input_dim // 4 ** len(channel_counts)
        for i in range(dense_count):
            self.dense_layers.append(nn.Linear(self.in_features, dense_nodes))

        if output_dim is not None:
            self.output_layer = nn.Linear(dense_nodes, output_dim)

    # Define forward pass
    def forward(self, x):

        outputs = []

        # Convolutional part
        for i, layer in enumerate(self.conv_layers):
            
            x = layer(x)

            # Record shortcut
            if i % 3 == 0:
                downsampled = x

            x = self.activation(x)
            x = nn.functional.dropout2d(x, p = self.dropout, training = self.training)

            # Re-add shortcut
            if i % 3 == 2:
                x += downsampled
                outputs.append(x)

        # Smush last two dimensions
        if len(self.dense_layers) > 0:
            x = x.permute(0, 3, 1, 2)
            x = x.contiguous().view(1, -1, self.in_features)
            x = nn.functional.dropout(x, p = self.dropout, training = self.training)

            # Fully conntected part
            for layer in self.dense_layers:
                x = self.activation(layer(x))

        if self.output_dim is not None:
            x = self.output_layer(x)
            x = x.permute(0, 2, 1)
        
        outputs.append(x)

        return outputs

def conv_layer(in_channels, out_channels, downsample = False):
    return nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 3,
        stride = 2 if downsample else 1,
        padding = 1,
    )

def conv_block(in_channels, out_channels):
    return [
        conv_layer(in_channels, out_channels, downsample = True),
        conv_layer(out_channels, out_channels),
        conv_layer(out_channels, out_channels),
    ]

