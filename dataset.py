from torch.utils.data import Dataset
from torchaudio import load
from pathlib import Path


class wavDataset(Dataset):
    """
    An extension of the pytorch Dataset for a generic wav Dataset.
    """

    def __init__(self, wavPath: Path | str, transform: callable = None, backend=None):
        """
        Arguments:
            wavPath (Path | str): Path to the audio dir with all the wav files
            transform (callable , optional): Optional transformation to be applied on a sample
            backend (str , optional): ["ffmpeg", "sox", "soundfile"]
        """
        self.wavPath = Path(wavPath)
        self.transform = transform
        self._wav_list = None
        self.backend = backend

    def __len__(self):
        """
        The method that returns the length of the dataset.
        It returns the number of wav files at wavPath with naive globbing"
        """
        return len(list(self.wavPath.glob("*.wav")))

    def __getitem__(self, idx):
        if self._wav_list is None:
            self._set_wav_files()
        wav_file_path = self._wav_files[idx]
        audio, sample_rate = load(wav_file_path, normalize=True, backend=None)
        if self.transform:
            audio = self.transform(audio)
        return (audio, sample_rate, wav_file_path)

    def _set_wav_files(self):
        """Retrieve and cache the sorted list of .wav files."""
        self._wav_files = sorted(list(self.wavPath.glob("*.wav")))
