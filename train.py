import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Importing the models and dataset class
from models.Generator import Generator
from models.Discriminator import Discriminator
from models.Dataset import TextureDataset
from config import config  

# Configuration
MODEL_SAVE_PATH = './models/checkpoints/'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
train_transform = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize datasets and loaders
train_dataset = TextureDataset(split='train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)

# Initialize models
generator = Generator(config['LATENT_DIM']).to(device)
discriminator = Discriminator().to(device)

# Initialize optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['LR'], betas=(config['BETA1'], config['BETA2']))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['LR'], betas=(config['BETA1'], config['BETA2']))

# Initialize gradient scalers
dScalar = GradScaler()
gScalar = GradScaler()

# Lists to store losses
d_losses = []
g_losses = []

# Training loop
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

    # Temporary lists to store losses for each batch
    temp_d_losses = []
    temp_g_losses = []

    for i, images in loop:
        real_images = images.to(device)

        ### Train Discriminator
        optimizer_D.zero_grad()
        with autocast():
            fake_images = generator(real_images)
            real_output = discriminator(real_images)
            d_real_loss = F.mse_loss(real_output, torch.ones_like(real_output))
            fake_images = torchvision.transforms.CenterCrop(256)(fake_images)
            fake_output = discriminator(fake_images.detach())
            d_fake_loss = F.mse_loss(fake_output, torch.zeros_like(fake_output))
            d_loss = (d_real_loss + d_fake_loss)

        dScalar.scale(d_loss).backward()
        dScalar.step(optimizer_D)
        dScalar.update()

        temp_d_losses.append(d_loss.item())

        ### Train Generator
        optimizer_G.zero_grad()
        with autocast():
            fake_output = discriminator(fake_images)
            g_loss = F.mse_loss(fake_output, torch.ones_like(fake_output))
            l1_loss = F.l1_loss(fake_images, real_images) * config['L1_LOSS_WEIGHT']
            g_final_loss = g_loss + l1_loss

        gScalar.scale(g_final_loss).backward()
        gScalar.step(optimizer_G)
        gScalar.update()

        temp_g_losses.append(g_final_loss.item())

        # Update the progress bar
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

    # Calculate average losses for the epoch and append to loss lists
    epoch_d_loss = sum(temp_d_losses) / len(temp_d_losses)
    epoch_g_loss = sum(temp_g_losses) / len(temp_g_losses)
    d_losses.append(epoch_d_loss)
    g_losses.append(epoch_g_loss)

    # Save the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), os.path.join(MODEL_SAVE_PATH, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(MODEL_SAVE_PATH, f'discriminator_epoch_{epoch+1}.pth'))
        print(f'Models saved at epoch {epoch+1}')

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()
