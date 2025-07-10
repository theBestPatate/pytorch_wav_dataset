from dataset import wavDataset
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

# Creating a raw dataset(audio in torch.tensor)
dataset = wavDataset("dataset")

# Creating a dataset of melspectrograms
mel_spec = MelSpectrogram(sample_rate=16000)
mel_dataset = wavDataset("dataset", transform=mel_spec)
# Notes: You can Compose transforms and create custom ones

dataset_loader = DataLoader(mel_dataset, batch_size=16, shuffle=True, num_workers=4)
