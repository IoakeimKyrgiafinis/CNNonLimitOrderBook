"""
CNN + Transformer model for limit order book price movement prediction.

Architecture:
    Input  (batch, seq_len=100, num_features=7)
    1D CNN block 1   : local patterns across feature dimension
    1D CNN block 2   : deeper compression
    Positional encoding
    Transformer encoder : self-attention across the 100 timesteps
    Global average pooling : collapse sequence to one vector
    Linear classification head
    Output (batch, 3)  : logits for {DOWN, FLAT, UP}

"""

import math
import torch
import torch.nn as nn


# Positional encoding 

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Transformers have no built-in sense of order, so we inject a fixed pattern
    that lets the model tell timestep 1 apart from timestep 99.
    """

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape (1, max_len, d_model) so it broadcasts over the batch dimension
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# CNN feature extractor 

class CNNBlock(nn.Module):
    """
    A single 1D convolution block operating along the TIME dimension.
    Input/output shape: (batch, channels, seq_len)

    Conv1d here treats each of the 7 features as a separate "channel" and
    slides a small window across time, picking up local temporal patterns
    (e.g. a sudden imbalance shift over 3-5 consecutive snapshots).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2   # keeps seq_len unchanged ("same" padding)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Full model 

class LOBTransformer(nn.Module):
    """
    CNN + Transformer model for 3-class LOB price movement prediction.

    Args:
        num_features:  number of input features per timestep (7 in our case)
        cnn_channels:  output channels of the CNN blocks (acts as d_model for the transformer)
        num_heads:     number of attention heads in the transformer encoder
        num_layers:    number of stacked transformer encoder layers
        ff_dim:        hidden dimension of the transformer's feed-forward sublayer
        num_classes:   3 (DOWN, FLAT, UP)
        dropout:       dropout probability used throughout
    """

    def __init__(
        self,
        num_features: int = 6,
        cnn_channels: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        #  CNN feature extractor 
        # Conv1d expects (batch, channels, seq_len), so we transpose on the way in.
        self.cnn1 = CNNBlock(in_channels=num_features, out_channels=cnn_channels, kernel_size=5)
        self.cnn2 = CNNBlock(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3)

        #  Positional encoding 
        self.pos_encoding = PositionalEncoding(d_model=cnn_channels)

        #  Transformer encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # so we keep (batch, seq_len, features) ordering throughout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #  Classification head 
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_features)  e.g. (32, 100, 7)
        returns: (batch, num_classes)      raw logits, NOT softmax-ed
                 (use CrossEntropyLoss during training, softmax only at inference)
        """

        #  CNN expects (batch, channels, seq_len) 
        x = x.transpose(1, 2)          # (batch, num_features, seq_len)
        x = self.cnn1(x)               # (batch, cnn_channels, seq_len)
        x = self.cnn2(x)               # (batch, cnn_channels, seq_len)
        x = x.transpose(1, 2)          # back to (batch, seq_len, cnn_channels)

        #  Transformer expects (batch, seq_len, d_model) 
        x = self.pos_encoding(x)
        x = self.transformer(x)        # (batch, seq_len, cnn_channels)

        #  Global average pooling across the time dimension 
        x = x.mean(dim=1)              # (batch, cnn_channels)

        #  Classification head 
        x = self.dropout(x)
        logits = self.classifier(x)    # (batch, num_classes)

        return logits


#  Quick sanity check 

if __name__ == "__main__":
    model = LOBTransformer()

    # fake batch: 8 sequences, 100 timesteps, 7 features
    dummy_input = torch.randn(8, 100, 6)
    logits = model(dummy_input)

    print(f"Input shape:     {dummy_input.shape}")
    print(f"Logits shape:    {logits.shape}")      # expect (8, 3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")