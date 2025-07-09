from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms, utils
from pathlib import Path


class wavDataset(Dataset):
    """
    An extension of the pytorch Dataset for a generic wav Dataset.
    """

    def __init__(self, wavPath: Path | str, transform: callable = None):
        """
        Arguments:
            wavPath (Path | str): Path to the audio dir with all the wav files
            transform (callable,optional): Optional transformation to be applied on a sample
        """
        self.wavPath = wavPath
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
