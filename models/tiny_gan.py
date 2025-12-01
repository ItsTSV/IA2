import torch.nn as nn


def init_c(layer):
    nn.init.normal_(layer.weight.data, 0.0, 0.02)
    return layer


def init_bn(layer):
    nn.init.normal_(layer.weight.data, 1.0, 0.02)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class DiscBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DiscBlock, self).__init__()
        self.block = nn.Sequential(
            # Depthwise Convolution -- reduce dimensions, filter channels separately
            init_c(nn.Conv2d(in_c, in_c, 4, 2, 1, groups=in_c, bias=False)),
            init_bn(nn.BatchNorm2d(in_c)),
            nn.LeakyReLU(0.2, inplace=True),

            # Pointwise Convolution -- mix channels
            init_c(nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)),
            init_bn(nn.BatchNorm2d(out_c)),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class GenBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(GenBlock, self).__init__()
        self.block = nn.Sequential(
            # Upsample
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Depthwise Convolution -- filter channels separately
            init_c(nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False)),
            init_bn(nn.BatchNorm2d(in_c)),
            nn.ReLU(True),

            # Pointwise Convolution -- mix channels
            init_c(nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)),
            init_bn(nn.BatchNorm2d(out_c)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class TinyGenerator(nn.Module):
    def __init__(self, latent_vector_size, feature_map_size, color_channels):
        super(TinyGenerator, self).__init__()
        fm = feature_map_size

        self.network = nn.Sequential(
            # Latent Vector to Feature Maps
            init_c(nn.ConvTranspose2d(latent_vector_size, 8 * fm, 4, 1, 0, bias=False)),
            init_bn(nn.BatchNorm2d(8 * fm)),
            nn.ReLU(True),

            # Upsampling Blocks
            GenBlock(8 * fm, 4 * fm),
            GenBlock(4 * fm, 2 * fm),
            GenBlock(2 * fm, fm),

            # Feature Maps to Image
            init_c(nn.Conv2d(fm, color_channels, 3, 1, 1, bias=False)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class TinyDiscriminator(nn.Module):
    def __init__(self, color_channels, feature_map_size):
        super(TinyDiscriminator, self).__init__()
        fm = feature_map_size

        self.network = nn.Sequential(
            # Image to feature maps
            init_c(nn.Conv2d(color_channels, fm, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # Downsampling Blocks
            DiscBlock(fm, 2 * fm),
            DiscBlock(2 * fm, 4 * fm),

            # Feature maps to prediction
            init_c(nn.Conv2d(4 * fm, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
