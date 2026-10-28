
import torch
import torch.nn as nn


# Input size and number of classes for each dataset.
TASKS = {
    "mnist":      {"in_features": 28 * 28,     "num_classes": 10},
    "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10},
    "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000},
}


class Block(nn.Module):
    """Residual block with normalization and gated activation."""

    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.gate = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Normalize and apply the gated activation.
        h = self.norm(x)
        value = self.fc1(h)
        gate = nn.functional.silu(self.gate(h))
        h = value * gate
        h = self.dropout(h)
        h = self.fc2(h)

        # Add the original input through the skip connection.
        return x + h


class ResidualMLP(nn.Module):
    """Residual MLP with optional image patches and spatial mixing."""

    def __init__(self, in_features, num_classes, width=128, hidden=512, depth=30):
        super().__init__()

        # Use patches for CIFAR-10 and ImageNet32.
        self.use_patches = (in_features == 3 * 32 * 32)

        if self.use_patches:
            self.patch_size = 8
            self.num_patches = 16

            self.patch_embed = nn.Linear(192, width)

            self.patch_norm = nn.LayerNorm(width)
            self.spatial_mix = nn.Linear(16,16)

            self.patch_output = nn.Linear(16 * width, width)

        else:
            # MNIST uses flattened image inputs.
            self.input_layer = nn.Linear(in_features, width)

        # Create the residual blocks.
        self.blocks = nn.ModuleList([Block(width, hidden) for _ in range(depth)])

        # Classification head.
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)
        self.head_dropout = nn.Dropout(p=0.2)

        # Initialize weights and biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Return raw class scores (logits)."""

        if self.use_patches:
            # Split images into 16 patches of size 8x8.
            x = x.reshape(-1, 3, 32, 32)

            x = x.unfold(2, 8, 8).unfold(3, 8, 8)
            x = x.permute(0, 2, 3, 1, 4, 5)
            x = x.reshape(-1, 16, 192)

            # Embed patches and mix spatial information.
            x = self.patch_embed(x)

            h = self.patch_norm(x)
            h = self.spatial_mix(h.transpose(1, 2)).transpose(1, 2)
            x = x + h

            x = self.patch_output(x.flatten(start_dim=1))
        else:
            # Flatten MNIST images and project to model width.
            x = x.flatten(start_dim=1)
            x = self.input_layer(x)

        # Pass through the residual backbone.
        for block in self.blocks:
            x = block(x)

        # Produce final class scores.
        x = self.head_fc1(self.head_norm(x))
        x = nn.functional.gelu(x)
        x = self.head_dropout(x)
        x = self.head_fc2(x)

        return x

    def predict(self, x):
        """Return class probabilities for inference."""
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)
