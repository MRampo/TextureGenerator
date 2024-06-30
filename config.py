config = {
    'DATASET_PATH': './path/to/dataset',  # Path to the dataset
    'BATCH_SIZE': 16,                    # Batch size for training
    'LATENT_DIM': 100,                   # Latent dimension for the generator
    'LR': 0.0002,                        # Learning rate for the optimizers
    'BETA1': 0.5,                        # Beta1 hyperparameter for Adam optimizer
    'BETA2': 0.999,                      # Beta2 hyperparameter for Adam optimizer
    'L1_LOSS_WEIGHT': 100,               # Weight for the L1 loss in the generator
    'NUM_EPOCHS': 10                     # Number of epochs for training
}

# Most of the values for these parameteres have been taken from the original study mentioned in the paper
