import torch
import torch.nn as nn

# Define the Residual Block
class ResidualBlock(nn.Module):
    """
    Residual Block used in the Generator model.

    Args:
        channels (int): Number of input and output channels for the block.
    """
    def __init__(self, channels):
        """
        Initialize the Residual Block.

        Args:
            channels: The number of input and output channels for the block.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),  
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        """
        Forward pass for the Residual Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with residual connection.
        """
        # Return the input with the residual connection added to the output of the block
        return x + self.layers(x)

# Define the Generator (SRResNet)
class Generator(nn.Module):
    """
    Generator model for Super-Resolution (SRResNet).

    Args:
        base_channels (int): Number of channels in the first convolutional layer.
        n_ps_blocks (int): Number of PixelShuffle blocks.
        n_res_blocks (int): Number of Residual blocks.
    """
    def __init__(self, base_channels=64, n_ps_blocks=1, n_res_blocks=16):
        """
        Initialize the Generator (SRResNet) model.

        Args:
            base_channels: The number of channels in the first convolutional layer.
            n_ps_blocks: The number of PixelShuffle blocks.
            n_res_blocks: The number of Residual blocks.
        """
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=9, padding=4),
            nn.SiLU(),  
        )
        res_blocks = [ResidualBlock(base_channels) for _ in range(n_res_blocks)]
        res_blocks += [
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)
        ps_blocks = []
        for _ in range(n_ps_blocks):
            ps_blocks += [
                nn.Conv2d(base_channels, 4 * base_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(),  
            ]
        self.ps_blocks = nn.Sequential(*ps_blocks)
        self.out_layer = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Forward pass for the Generator model.

        Args:
            x (torch.Tensor): Low-resolution input tensor.

        Returns:
            torch.Tensor: High-resolution output tensor.
        """
        x_res = self.in_layer(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.out_layer(x)
        return x