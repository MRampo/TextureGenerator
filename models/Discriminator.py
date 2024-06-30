import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Define the model as a sequence of layers
        self.model = nn.Sequential(
            # First convolutional layer: input channels = 3 (RGB image), output channels = 64, kernel size = 4, stride = 2, padding = 1
            # This layer reduces the spatial dimensions from 256x256 to 128x128
            nn.Conv2d(3, 64, 4, 2, 1),  # Output: 64x128x128
            nn.LeakyReLU(0.2, inplace=True),  # Apply LeakyReLU activation with a negative slope of 0.2

            # Second convolutional layer: input channels = 64, output channels = 128, kernel size = 4, stride = 2, padding = 1
            # This layer reduces the spatial dimensions from 128x128 to 64x64
            nn.Conv2d(64, 128, 4, 2, 1),  # Output: 128x64x64
            nn.BatchNorm2d(128),  # Apply batch normalization to stabilize training
            nn.LeakyReLU(0.2, inplace=True),  # Apply LeakyReLU activation

            # Third convolutional layer: input channels = 128, output channels = 256, kernel size = 4, stride = 2, padding = 1
            # This layer reduces the spatial dimensions from 64x64 to 32x32
            nn.Conv2d(128, 256, 4, 2, 1),  # Output: 256x32x32
            nn.BatchNorm2d(256),  # Apply batch normalization
            nn.LeakyReLU(0.2, inplace=True),  # Apply LeakyReLU activation

            # Fourth convolutional layer: input channels = 256, output channels = 512, kernel size = 4, stride = 2, padding = 1
            # This layer reduces the spatial dimensions from 32x32 to 16x16
            nn.Conv2d(256, 512, 4, 2, 1),  # Output: 512x16x16
            nn.BatchNorm2d(512),  # Apply batch normalization
            nn.LeakyReLU(0.2, inplace=True)  # Apply LeakyReLU activation
        )

        # Fully connected layer to convert the flattened feature map into a single output
        # The input size is adjusted for the 512x16x16 feature maps
        self.fc = nn.Linear(512 * 16 * 16, 1)  # Adjusted for 512x16x16 feature maps

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.model(x)
        # Flatten the features for each image in the batch
        x = torch.flatten(x, start_dim=1)
        # Pass the flattened features through the fully connected layer
        x = self.fc(x)  # Output: [batch_size, 1]
        # Apply sigmoid activation to ensure the output is a probability
        return torch.sigmoid(x)

