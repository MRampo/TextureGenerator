import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: 3x256x256
            nn.Conv2d(3, 64, 4, 2, 1),  # Output: 64x128x128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # Output: 128x64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # Output: 256x32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  # Output: 512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Adjust the fully connected layer to account for the new feature map size
        self.fc = nn.Linear(512 * 16 * 16, 1)  # Adjusted for 512x16x16 feature maps

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)  # Flatten the features for each image
        x = self.fc(x)  # Output: [batch_size, 1]
        return torch.sigmoid(x)  # Ensure output is a probability
