import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ImagePairsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset by setting the root directory and transform, and loading image pairs.

        Parameters:
        - root_dir (str): The root directory containing the image pairs.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self.load_image_pairs()  # Load the image pairs from the root directory

    def __len__(self):
        """
        Returns the total number of image pairs in the dataset.
        """
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Retrieves the image pair at the specified index.

        Parameters:
        - idx (int): Index of the image pair to retrieve.

        Returns:
        - Tuple of input and target images (transformed if applicable).
        """
        input_img_path, target_img_path = self.image_pairs[idx]  # Get the file paths for the input and target images
        input_img = Image.open(input_img_path).convert("RGB")  # Open and convert the input image to RGB
        target_img = Image.open(target_img_path).convert("RGB")  # Open and convert the target image to RGB

        if self.transform:
            # Apply the transformations if any
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img  # Return the transformed images as a tuple

    def load_image_pairs(self):
        """
        Loads and returns a list of tuples containing the file paths of input and target image pairs.

        Returns:
        - List of tuples: Each tuple contains the file paths of an input image and its corresponding target image.
        """
        image_pairs = []  # Initialize an empty list to store image pairs
        for root, _, files in os.walk(self.root_dir):  # Walk through the directory
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):  # Check for image files
                    input_path = os.path.join(root, file)  # Get the full path of the input image
                    target_path = input_path.replace("input", "target")  # Replace 'input' with 'target' to get the target image path
                    if os.path.exists(target_path):  # Check if the target image exists
                        image_pairs.append((input_path, target_path))  # Append the pair of paths to the list
        return image_pairs  # Return the list of image pairs
