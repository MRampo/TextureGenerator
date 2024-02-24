import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: 3x512x512
            # First Down-sampling layer: 3x512x512 -> 64x256x256
            # Second Down-sampling layer: 64x256x256 -> 128x128x128
            # Third Down-sampling layer: 128x128x128 -> 256x64x64
            # Fourth Down-sampling layer: 256x64x64 -> 512x32x32
            # Final layer: 512x32x32 -> 
            
            nn.Conv2d(3, 64, 4, 2, 1),  
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Final layer to get a single output value
            nn.Conv2d(512, 1, 4, 1, 0),  
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        # Flatten the output for binary classification
        return output.view(-1, 1).squeeze(1)

