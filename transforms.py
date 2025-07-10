from torch import Tensor
from torch.nn import ZeroPad1d


class audio_rescale(object):
    """
    Trims or pads audio waveforms to a specified duration.

    Attributes:
        final_duration (int | float): Desired final duration in seconds.
        sr (int): Sample rate (default: 16000 Hz).
        waveform (Tensor): Audio waveform tensor.
        num_samples (int): Number of samples in the waveform.
        padding_scheme (int | list[int]): Padding amounts before and after the waveform.
    """

    def __init__(
        self,
        final_duration: int | float,
        sr: int = 16000,
        padding_scheme: int | list[int] = [0, 1],
    ):
        """
        Initializes the audio_rescale object.

        Args:
            final_duration (int | float): Final duration in seconds.
            sr (int, optional): Sample rate (default: 16000 Hz).
            padding_scheme (int | list[int], optional): Padding scheme (default: [0, 1]).
        """
        assert isinstance(final_duration, (int, float)), (
            "final_duration must be an int or float."
        )
        self.final_duration: int | float = final_duration
        self.sr: int = sr
        self.waveform: Tensor = None
        self.num_samples: int = 0
        self.padding_scheme: list[int] = (
            padding_scheme
            if isinstance(padding_scheme, list)
            else [padding_scheme, padding_scheme]
        )

    def __call__(self, sample: Tensor) -> Tensor:
        """
        Processes the input audio sample.

        Args:
            sample (Tensor): Input audio waveform tensor.

        Returns:
            Tensor: Trimmed or padded audio waveform tensor.
        """
        self.waveform = sample
        self.num_samples = len(sample)
        return self.pad_trim(self.final_duration)

    def pad_trim(self, seconds: int | float) -> Tensor:
        """
        Trims or pads the waveform to the specified duration.

        Args:
            seconds (int | float): Target duration in seconds.

        Returns:
            Tensor: Trimmed or padded audio waveform tensor.
        """
        num_samples = int(self.sr * seconds)
        if self.num_samples > num_samples:
            return self.waveform[:num_samples]
        elif self.num_samples < num_samples:
            padding_amount = num_samples - self.num_samples
            self.padding_scheme[1] = padding_amount
            return ZeroPad1d(tuple(self.padding_scheme))(self.waveform)
        else:
            return self.waveform
