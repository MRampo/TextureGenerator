import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ImagePairsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self.load_image_pairs()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_img_path, target_img_path = self.image_pairs[idx]
        input_img = Image.open(input_img_path).convert("RGB")
        target_img = Image.open(target_img_path).convert("RGB")

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

    def load_image_pairs(self):
        image_pairs = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    input_path = os.path.join(root, file)
                    target_path = input_path.replace("input", "target")  # Assuming your input and target images are in separate folders
                    if os.path.exists(target_path):
                        image_pairs.append((input_path, target_path))
        return image_pairs