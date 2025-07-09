from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms, utils
from pathlib import Path


class wavDataset(Dataset):
    """
    An extension of the pytorch Dataset for a generic wav Dataset.
    """

    def __init__(self, wavPath: Path, transform: callable = None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
