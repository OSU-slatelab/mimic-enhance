import torch
import torch.nn as nn

class AECNN(nn.Module):

    def __init__(
        self,
        channel_counts = [64, 128, 256],
        kernel_size = 11,
        block_size = 3,
        activation = lambda x: nn.functional.leaky_relu(x, negative_slope = 0.3),
        dropout = 0.2,
        training = True,
    ):
        super(AECNN, self).__init__()

        # Store hyperparameters
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.dropout = dropout
        self.activation = activation

        # Initialize all layer containers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # Encoder uses conv layers with stride to downsample inputs
        in_channels = 1
        for out_channels in channel_counts:
            for i in range(block_size):
                self.encoder_layers.append(self.conv_layer(in_channels, out_channels, downsample = in_channels != 1))
                in_channels = out_channels

        # Decoder layers get concatenated with corresponding encoder layers (unet-style)
        in_channels = None
        for out_channels in reversed(channel_counts):
            for i in range(block_size):
                if in_channels is None:
                    in_channels = out_channels
                else:
                    self.decoder_layers.append(self.conv_layer(in_channels, out_channels))
                    in_channels = out_channels * 2

        # Final layer doesn't change size, just filters
        self.decrease_channels = self.conv_layer(in_channels, out_channels = 1)


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

        # Apply encoder downsampling layers
        encoder_outputs = [x]
        for i, layer in enumerate(self.encoder_layers):
            encoder_outputs.append(self.activation(layer(encoder_outputs[-1])))

            # dropout once each block
            if i % self.block_size == 0:
                encoder_outputs[-1] = nn.functional.dropout(encoder_outputs[-1], p = self.dropout)

        # Apply upsampling and decoder layers
        decoder_inputs = encoder_outputs[-1]
        for i, layer in enumerate(self.decoder_layers):
            decoder_inputs = nn.functional.interpolate(decoder_inputs, scale_factor = 2)
            decoder_output = self.activation(layer(decoder_inputs))

            # Concatenate layer with corresponding encoder layer
            decoder_inputs = torch.cat((encoder_outputs[-i - 2], decoder_output), dim = 1)
            
            # Dropout once each block
            if i % self.block_size == 0:
                decoder_inputs = nn.functional.dropout(decoder_inputs, p = self.dropout)

        # Reduce to 1 channel and scale from -1 to 1
        return torch.tanh(self.decrease_channels(decoder_inputs))
