import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=300, hidden_dim1=256, hidden_dim2=128, bottleneck_dim=30):
        """
        AutoEncoder with two hidden layers for dimensionality reduction of word embeddings.

        Args:
            input_dim (int): Dimension of the input embeddings (default: 300).
            hidden_dim1 (int): Dimension of the first hidden layer (default: 256).
            hidden_dim2 (int): Dimension of the second hidden layer (default: 128).
            bottleneck_dim (int): Dimension of the bottleneck layer (output dimension, default: 30).
        """
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, bottleneck_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )

        # Store bottleneck output
        self.bottleneck_output = None

    def forward(self, x):
        """
        Forward pass through the AutoEncoder.

        Args:
            x (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Reconstructed embeddings.
        """
        self.bottleneck_output = self.encoder(x)  # Save bottleneck output
        decoded = self.decoder(self.bottleneck_output)
        return decoded