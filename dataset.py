from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms, utils


class wavDataset(Dataset):
    """
    An extension of the pytorch Dataset for a generic wav Dataset.
    """

    def __init__(self, wavPath, transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
