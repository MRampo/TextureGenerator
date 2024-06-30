import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        # Define the encoder as a sequence of convolutional layers
        self.encoder = nn.Sequential(
            #  input channels = 3 (RGB image), output channels = 32
            # Kernel size = 4, stride = 2, padding = 1
            # This layer reduces the spatial dimensions from 256x256 to 128x128
            nn.Conv2d(3, 32, 4, 2, 1),  # Output: 32x128x128
            nn.BatchNorm2d(32),  # Apply batch normalization to stabilize training
            nn.LeakyReLU(0.2, inplace=True),  # Apply LeakyReLU activation with a negative slope of 0.2

            # input channels = 32, output channels = 64
            # Reduces the spatial dimensions from 128x128 to 64x64
            nn.Conv2d(32, 64, 4, 2, 1),  # Output: 64x64x64
            nn.BatchNorm2d(64),  # Apply batch normalization
            nn.LeakyReLU(0.2, inplace=True),  # Apply LeakyReLU activation

            # input channels = 64, output channels = 128
            # Reduces the spatial dimensions from 64x64 to 32x32
            nn.Conv2d(64, 128, 4, 2, 1),  # Output: 128x32x32
            nn.BatchNorm2d(128),  # Apply batch normalization
            nn.LeakyReLU(0.2, inplace=True),  # Apply LeakyReLU activation

            # input channels = 128, output channels = 256
            # Reduces the spatial dimensions from 32x32 to 16x16
            nn.Conv2d(128, 256, 4, 2, 1),  # Output: 256x16x16
            nn.BatchNorm2d(256),  # Apply batch normalization
            nn.LeakyReLU(0.2, inplace=True),  # Apply LeakyReLU activation

            # input channels = 256, output channels = latent_dim
            # Maintains the spatial dimensions at 16x16
            nn.Conv2d(256, latent_dim, 3, 1, 1),  # Output: latent_dim x 16 x 16
            nn.BatchNorm2d(latent_dim),  # Apply batch normalization
            nn.LeakyReLU(0.2, inplace=True)  # Apply LeakyReLU activation
        )
        
    def forward(self, x):
        # Pass the input through the encoder network
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        # Define the decoder as a sequence of transposed convolutional layers
        self.decoder = nn.Sequential(
            # input channels = latent_dim, output channels = 512
            # Kernel size = 4, stride = 2, padding = 1
            # This layer increases the spatial dimensions from 16x16 to 32x32
            nn.ConvTranspose2d(latent_dim, 512, 4, 2, 1),  # Output: 512x32x32
            nn.BatchNorm2d(512),  # Apply batch normalization
            nn.ReLU(True),  # Apply ReLU activation

            # input channels = 512, output channels = 256
            # Increases the spatial dimensions from 32x32 to 64x64
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Output: 256x64x64
            nn.BatchNorm2d(256),  # Apply batch normalization
            nn.ReLU(True),  # Apply ReLU activation

            # input channels = 256, output channels = 128
            # Increases the spatial dimensions from 64x64 to 128x128
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # Output: 128x128x128
            nn.BatchNorm2d(128),  # Apply batch normalization
            nn.ReLU(True),  # Apply ReLU activation

            # input channels = 128, output channels = 64
            # Increases the spatial dimensions from 128x128 to 256x256
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # Output: 64x256x256
            nn.BatchNorm2d(64),  # Apply batch normalization
            nn.ReLU(True),  # Apply ReLU activation

            #  input channels = 64, output channels = 3 (RGB image)
            # Increases the spatial dimensions from 256x256 to 512x512
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # Output: 3x512x512
            nn.Tanh()  # Apply Tanh activation to scale the output to the range [-1, 1]
        )
        
    def forward(self, x):
        # Pass the input through the decoder network
        return self.decoder(x)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        # Initialize the encoder and decoder with the specified latent dimension
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        # Pass the input image through the encoder to obtain the latent representation
        latent_code = self.encoder(x)
        # Pass the latent representation through the decoder to reconstruct the image
        reconstructed_img = self.decoder(latent_code)
        # Return the reconstructed image
        return reconstructed_img
