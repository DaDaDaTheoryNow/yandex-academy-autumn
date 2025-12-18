import torch

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_best_device()

IMG_SIZE = 196
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3

DATASET_ZIP = 'dataset.zip'
BEST_MODEL_PATH = 'release-weights.pth'
RESUME_PATH = 'release-weights.pth'
OUTPUT_CSV = 'predictions.csv'
