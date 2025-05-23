from typing import Any

import obspy
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from seisbench.models.base import WaveformModel


class QuakeXNet(WaveformModel):
    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["stride"] = (_annotate_args["stride"][0], 1000)
    _annotate_args["threshold"] = ("Detection threshold for non-noise class", 0.8)

    def __init__(
        self,
        sampling_rate=50,
        classes=4,
        output_type="point",
        labels=["eq", "px", "no", "su"],
        pred_sample=19,
        num_channels=3,
        num_classes=4,
        dropout_rate=0.4,
        filter_kwargs={
            "type": "bandpass",
            "freqmin": 1,
            "freqmax": 19.9,
            "corners": 4,
            "zerophase": True,
        },
        **kwargs,
    ):

        citation = (
            "Kharita, Akash, Marine Denolle, Alexander Hutko, and J. Renate Hartog."
            " A comprehensive machine learning and deep learning exploration for seismic event classification in the Pacific Northwest."
            " AGU24 (2024)."
        )

        super().__init__(
            citation=citation,
            output_type="point",
            component_order="ENZ",
            in_samples=5000,
            pred_sample=pred_sample,
            labels=labels,
            sampling_rate=sampling_rate,
            filter_kwargs=filter_kwargs,
            **kwargs,
        )

        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(3, 3), stride=2, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(64)

        # Calculate the input size for the fully connected layer dynamically
        self.fc_input_size = self._get_conv_output_size(num_channels, (129, 38))
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2_bn = nn.BatchNorm1d(num_classes)

        # Define dropout
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv_output_size(self, num_channels, input_dims):
        # Forward pass a dummy input through the conv layers to get the output size
        dummy_input = torch.randn(1, num_channels, *input_dims)
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = self.pool1(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool1(F.relu(self.bn4(self.conv4(x))))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = F.relu(self.bn7(self.conv7(x)))
        return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # output size: (8, 129, 38)
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))  # output size: (8, 64, 19)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))  # output size: (16, 64, 19)
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))  # output size: (16, 32, 10)
        x = self.dropout(x)

        x = F.relu(self.bn5(self.conv5(x)))  # output size: (32, 32, 10)
        x = F.relu(self.bn6(self.conv6(x)))  # output size: (32, 16, 5)
        x = self.dropout(x)

        x = F.relu(self.bn7(self.conv7(x)))  # output size: (64, 16, 5)

        x = x.view(x.size(0), -1)  # Flatten before fully connected layer
        x = self.dropout(x)

        x = F.relu(self.fc1_bn(self.fc1(x)))  # classifier
        x = self.fc2_bn(self.fc2(x))  # classifier

        return x

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch.cpu()

        # Detrend each component
        batch = linear_detrend(batch)

        # Create a Tukey window using scipy
        tukey_window = scipy.signal.windows.tukey(batch.shape[-1], alpha=0.1)

        # Apply the Tukey window to the batch
        batch *= tukey_window  # Broadcasting over last axis

        # Normalize each component by the standard deviation of their absolute values
        batch_abs = torch.abs(batch)
        batch /= batch_abs.std(dim=-1, keepdim=True) + 1e-10  # Avoid division by zero

        # Convert the processed waveforms to spectrograms
        spec = self.extract_spectrograms(batch, fs=argdict["sampling_rate"])

        return spec

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        return torch.softmax(batch, dim=-1)

    def classify_aggregate(self, annotations, argdict) -> list:
        try:
            assert len(annotations) == 4
        except:
            return []

        t = [annotations[0].stats.starttime + i for i in annotations[0].times()]
        onsets = get_onset_time(
            annotations.select(channel=f"{self.__class__.__name__}_no")[0],
            self._annotate_args["threshold"][1],
        )
        eq = annotations.select(channel=f"{self.__class__.__name__}_eq")[0].data
        px = annotations.select(channel=f"{self.__class__.__name__}_px")[0].data
        su = annotations.select(channel=f"{self.__class__.__name__}_su")[0].data
        return [
            {"start": t[i[0]], "eq": eq[i[0]], "px": px[i[0]], "su": su[i[0]]}
            for i in onsets
        ]

    @staticmethod
    def extract_spectrograms(waveforms, fs, nperseg=256, overlap=0.5):
        """
        Extract spectrograms from waveforms using PyTorch operations,
        but mimicking the behavior of scipy.signal.spectrogram for 
        compatibility with pre-trained weights.

        Parameters:
            waveforms: Tensor of shape (n_waveforms, n_channels, n_samples)
            fs: Sampling rate (Hz)
            nperseg: Number of points per FFT segment
            overlap: Fractional overlap between segments

        Returns:
            spectrograms: Tensor of shape (n_waveforms, n_channels, frequencies, time_segments)
        """
        # Calculate overlap parameters
        noverlap = int(nperseg * overlap)
        hop_length = nperseg - noverlap
        
        # Get shapes
        batch_size, n_channels, n_samples = waveforms.shape
        
        # Create window - use Hann window to match scipy default
        window = torch.hann_window(nperseg, device=waveforms.device)
        
        # Process one sample from the batch to determine output dimensions
        # Run scipy.signal.spectrogram for reference
        import numpy as np
        import scipy.signal as signal
        
        sample_np = waveforms[0, 0].cpu().detach().numpy()
        
        try:
            f, t, Sxx_ref = signal.spectrogram(
                sample_np, fs=fs, nperseg=nperseg, noverlap=noverlap
            )
            n_freqs, n_times = Sxx_ref.shape
        except Exception as e:
            print(f"Error in reference spectrogram computation: {e}")
            print("Falling back to default sizes")
            # Typical sizes for 5000 samples with nperseg=256, noverlap=128
            n_freqs = nperseg // 2 + 1  # For real signals, this is the number of unique frequencies
            n_times = (n_samples - noverlap) // (nperseg - noverlap)
        
        # Initialize output tensor
        spectrograms = torch.zeros(
            (batch_size, n_channels, n_freqs, n_times), 
            dtype=torch.float32, 
            device=waveforms.device
        )
        
        # Calculate spectrograms using PyTorch STFT
        for b in range(batch_size):
            for c in range(n_channels):
                # Get the waveform segment
                x = waveforms[b, c]
                
                # Handle potential NaN or Inf values
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning: NaN or Inf values detected in waveform at batch {b}, channel {c}")
                    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Apply STFT
                try:
                    stft_result = torch.stft(
                        x, 
                        n_fft=nperseg, 
                        hop_length=hop_length, 
                        win_length=nperseg,
                        window=window,
                        center=False,  # Match scipy default
                        normalized=False,  # Match scipy default
                        onesided=True,  # Match scipy default for real inputs
                        return_complex=True
                    )
                    
                    # Convert to power spectrogram (magnitude squared)
                    # Note: scipy.signal.spectrogram uses |STFT|²
                    power = torch.abs(stft_result)**2
                    
                    # Scale by window correction factor to match scipy's output
                    # scipy corrects by 1/(sum(window)^2)
                    scale = 1.0 / (torch.sum(window)**2)
                    power *= scale
                    
                    # Store in output tensor
                    if power.shape[0] <= n_freqs and power.shape[1] <= n_times:
                        spectrograms[b, c, :power.shape[0], :power.shape[1]] = power
                    else:
                        # Truncate if larger
                        spectrograms[b, c] = power[:n_freqs, :n_times]
                        
                except Exception as e:
                    print(f"Error in STFT computation at batch {b}, channel {c}: {e}")
                    # Leave as zeros in case of error
        
        return spectrograms


class QuakeXNetoneD(WaveformModel):
    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["stride"] = (_annotate_args["stride"][0], 2500)
    _annotate_args["threshold"] = ("Detection threshold for non-noise class", 0.5)

    def __init__(
        self,
        sampling_rate=50,
        classes=4,
        output_type="point",
        labels=["eq", "px", "no", "su"],
        pred_sample=19,
        num_channels=3,
        num_classes=4,
        dropout_rate=0.4,
        filter_kwargs={
            "type": "bandpass",
            "freqmin": 1,
            "freqmax": 19.9,
            "corners": 4,
            "zerophase": True,
        },
        **kwargs,
    ):

        citation = (
            "Kharita, Akash, Marine Denolle, Alexander Hutko, and J. Renate Hartog."
            " A comprehensive machine learning and deep learning exploration for seismic event classification in the Pacific Northwest."
            " AGU24 (2024)."
        )

        super().__init__(
            citation=citation,
            output_type="point",
            component_order="ENZ",
            in_samples=5000,
            pred_sample=pred_sample,
            labels=labels,
            sampling_rate=sampling_rate,
            filter_kwargs=filter_kwargs,
            **kwargs,
        )

        # Define the layers of the CNN architecture
        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=8, kernel_size=9, stride=1, padding=4
        )
        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=8, kernel_size=9, stride=2, padding=4
        )
        self.conv3 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3
        )
        self.conv4 = nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=7, stride=2, padding=3
        )
        self.conv5 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.conv6 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2
        )
        self.conv7 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(64)

        # Dynamically calculate the size of the first fully connected layer
        self.fc_input_size = self._get_conv_output_size(num_channels, input_length=5000)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2_bn = nn.BatchNorm1d(num_classes)

        # Define dropout
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv_output_size(self, num_channels, input_length):
        # Forward pass a dummy input through the conv layers to get the output size
        dummy_input = torch.randn(1, num_channels, input_length)
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = self.pool1(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool1(F.relu(self.bn4(self.conv4(x))))
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.pool1(F.relu(self.bn6(self.conv6(x))))
            x = F.relu(self.bn7(self.conv7(x)))
        return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool1(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = x.view(x.size(0), -1)  # Flatten before fully connected layer
        x = self.dropout(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2_bn(self.fc2(x))
        return x

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch.cpu()

        # Detrend each component
        batch = linear_detrend(batch)

        # Create a Tukey window using scipy
        tukey_window = scipy.signal.windows.tukey(batch.shape[-1], alpha=0.1)

        # Apply the Tukey window to the batch
        batch *= tukey_window  # Broadcasting over last axis

        # Normalize each component by the standard deviation of their absolute values
        batch_abs = torch.abs(batch)
        batch /= batch_abs.std(dim=-1, keepdim=True) + 1e-10  # Avoid division by zero

        return batch.float()

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        return torch.softmax(batch, dim=-1)

    def classify_aggregate(self, annotations, argdict) -> list:
        try:
            assert len(annotations) == 4
        except:
            return []

        t = [annotations[0].stats.starttime + i for i in annotations[0].times()]
        onsets = get_onset_time(
            annotations.select(channel=f"{self.__class__.__name__}_no")[0],
            self._annotate_args["threshold"][1],
        )
        eq = annotations.select(channel=f"{self.__class__.__name__}_eq")[0].data
        px = annotations.select(channel=f"{self.__class__.__name__}_px")[0].data
        su = annotations.select(channel=f"{self.__class__.__name__}_su")[0].data
        return [
            {"start": t[i[0]], "eq": eq[i[0]], "px": px[i[0]], "su": su[i[0]]}
            for i in onsets
        ]


def get_onset_time(trace: obspy.Trace, threshold: float) -> list:
    triggers = obspy.signal.trigger.trigger_onset(
        1 - trace.data, threshold, threshold / 2
    )
    return triggers


def linear_detrend(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply linear detrending similar to ObsPy.
    """
    # Time indices
    time = torch.arange(tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)

    # Calculate linear fit coefficients using least squares
    time_mean = time.mean()
    time_variance = ((time - time_mean) ** 2).sum()
    slope = ((tensor * (time - time_mean)).sum(dim=-1, keepdim=True)) / time_variance
    intercept = tensor.mean(dim=-1, keepdim=True) - slope * time_mean

    # Compute the trend
    trend = slope * time + intercept

    # Remove the trend from the original tensor
    return tensor - trend
