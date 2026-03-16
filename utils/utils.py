import torch

# set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# get device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")