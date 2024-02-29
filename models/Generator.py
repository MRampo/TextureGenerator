import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),     # Input: 3x256x256, Output: 32x128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),    # Output: 64x64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),   # Output: 128x32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # Output: 256x16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Adjusted to maintain 16x16 output
            nn.Conv2d(256, latent_dim, 3, 1, 1), # Output: latent_dim x 16 x 16
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 2, 1), # Input: latent_dim x 16 x 16, Output: 512x32x32
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),       # Output: 256x64x64
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),       # Output: 128x128x128
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),        # Output: 64x256x256
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),          # Output: 3x512x512
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.decoder(x)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.skip = nn.ConvTranspose2d(latent_dim, latent_dim, 4, 2, 1)
        
    def forward(self, x):
        latent_code = self.encoder(x)
        combined_code = latent_code 
        reconstructed_img = self.decoder(combined_code)
        return reconstructed_img

