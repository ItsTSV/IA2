import torch
import torch.nn as nn


class MyVisionTransformer(nn.Module):
    def __init__(
        self, patch_size=16, embedding_length=768, img_size=224, layer_count=3
    ):
        super(MyVisionTransformer, self).__init__()

        # Parameters
        self.patch_size = patch_size
        self.embedding_length = embedding_length
        self.patch_count = (img_size // patch_size) ** 2

        # Split img into patches
        self.patch_layer = nn.Conv2d(
            3, embedding_length, kernel_size=patch_size, stride=patch_size
        )

        # CLS token + Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_length))
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.patch_count + 1, embedding_length)
        )

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_length,
            nhead=12,
            dim_feedforward=embedding_length * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layer_count)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embedding_length), nn.Linear(embedding_length, 1)
        )

    def forward(self, x):
        # Get patches
        x = self.patch_layer(x)
        x = x.flatten(start_dim=2, end_dim=-1).transpose(1, 2)

        # Token + embeddings
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding

        # Transformer
        x = self.transformer(x)

        # Classify
        x = x[:, 0, :]
        predicted = self.classification_head(x)

        return predicted

    def load(self, path):
        self.load_state_dict(
            torch.load(
                path,
                weights_only=True,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self.eval()
