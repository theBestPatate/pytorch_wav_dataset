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
        self.wavPath = Path(wavPath)
        self.transform = transform
        self._wav_list = None

    def __len__(self):
        """
        The method that returns the length of the dataset.
        It returns the number of wav files at wavPath with naive globbing"
        """
        return len(self.wavPath.glob("*.wav"))

    def __getitem__(self, idx):
        if self._wav_list is None:
            self._set_wav_files()
        return len(self._wav_list)

    def _set_wav_files(self):
        """Retrieve and cache the list of .wav files."""
        self._wav_files = list(self.wavPath.glob("*.wav"))
