import torch.nn as nn

# Initialize weights according to the paper
def init_c(layer):
    nn.init.normal_(layer.weight.data, 0.0, 0.02)
    return layer

def init_bn(layer):
    nn.init.normal_(layer.weight.data, 1.0, 0.02)
    nn.init.constant_(layer.bias.data, 0)
    return layer


# Generator for mapping latent vector to img
class Generator(nn.Module):
    def __init__(self, latent_vector_size, feature_map_size, color_channels):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            # Latent vector to feature map
            nn.ConvTranspose2d(latent_vector_size, 8 * feature_map_size, 4, 1, 0, bias=False),
            init_bn(nn.BatchNorm2d(8 * feature_map_size)),
            nn.ReLU(True),
            # Blocks
            nn.ConvTranspose2d(8 * feature_map_size, 4 * feature_map_size, 4, 2, 1, bias=False),
            init_bn(nn.BatchNorm2d(4 * feature_map_size)),
            nn.ReLU(True),
            nn.ConvTranspose2d(4 * feature_map_size, 2 * feature_map_size, 4, 2, 1, bias=False),
            init_bn(nn.BatchNorm2d(2 * feature_map_size)),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * feature_map_size, feature_map_size, 4, 2, 1, bias=False),
            init_bn(nn.BatchNorm2d(feature_map_size)),
            nn.ReLU(True),
            # Feature map to img
            nn.ConvTranspose2d(feature_map_size, color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


# Discriminator for guessing whether the image is real or fake
class Discriminator(nn.Module):
    def __init__(self, color_channels, feature_map_size):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            # 3 channel input to feature maps
            init_c(nn.Conv2d(color_channels, feature_map_size, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # Convolutional blocks
            init_c(nn.Conv2d(feature_map_size, 2 * feature_map_size, 4, 2, 1, bias=False)),
            init_bn(nn.BatchNorm2d(2 * feature_map_size)),
            nn.LeakyReLU(0.2, inplace=True),
            init_c(nn.Conv2d(2 * feature_map_size, 4 * feature_map_size, 4, 2, 1, bias=False)),
            init_bn(nn.BatchNorm2d(4 * feature_map_size)),
            nn.LeakyReLU(0.2, inplace=True),
            init_c(nn.Conv2d(4 * feature_map_size, 8 * feature_map_size, 4, 2, 1, bias=False)),
            init_bn(nn.BatchNorm2d(8 * feature_map_size)),
            nn.LeakyReLU(0.2, inplace=True),
            # Feature maps to prediction
            init_c(nn.Conv2d(8 * feature_map_size, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
