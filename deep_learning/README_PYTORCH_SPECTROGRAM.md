# QuakeXNet with PyTorch Spectrogram Implementation

This README explains how to use the optimized QuakeXNet implementation with a PyTorch-based spectrogram calculation that is compatible with the pre-trained models.

## Background

The original QuakeXNet model was trained using spectrograms calculated with `scipy.signal.spectrogram`. When adapting the model to work with SeisBench, we need to ensure that the spectrogram calculation done in PyTorch produces outputs that are compatible with the pre-trained weights.

## Implementation Details

The key files are:

1. `classifier.py` - Contains the QuakeXNet model with an updated `extract_spectrograms` method that uses PyTorch operations but produces output similar to the scipy version.

2. `torch_spectrogram.py` - A standalone module that implements the PyTorch-based spectrogram function and includes validation code to ensure it produces results consistent with scipy.

3. `apply_quakexnet_demo.py` - A demonstration script showing how to apply the QuakeXNet model to seismic data using the optimized implementation.

## How to Use

### Single Event Processing

You can process a single seismic event with three components (Z, N, E) using the `apply_quakexnet_demo.py` script:

```bash
python apply_quakexnet_demo.py --z_file /path/to/Z.mseed --n_file /path/to/N.mseed --e_file /path/to/E.mseed --model trained_models/best_model_MyCNN_2d.pth
```

If you run the script without arguments, it will attempt to find a sample event in the data directory:

```bash
python apply_quakexnet_demo.py
```

### Batch Processing

You can process multiple events in batch mode:

```bash
python apply_quakexnet_demo.py --batch --data_dir /path/to/mseed/files --output results.csv
```

To process a specific event:

```bash
python apply_quakexnet_demo.py --batch --event_id uw10549638 --data_dir /path/to/mseed/files
```

### Using GPU Acceleration

To use GPU acceleration, add the `--device cuda` flag:

```bash
python apply_quakexnet_demo.py --device cuda
```

## How the Spectrogram Implementation Works

The PyTorch-based spectrogram implementation in `extract_spectrograms` method:

1. Uses PyTorch's `torch.stft` function to compute the Short-Time Fourier Transform
2. Applies a Hann window (matching scipy's default)
3. Computes the power spectrum by squaring the magnitude of the STFT
4. Applies a scale factor to match scipy's normalization

This ensures that the output spectrograms are as close as possible to those used during training, allowing the pre-trained weights to work correctly.

## Validation

You can validate that the PyTorch implementation matches the scipy implementation by running:

```bash
python torch_spectrogram.py
```

This will generate a test signal, compute spectrograms using both methods, and report the relative error between them.

## Model Performance

The models included in the `trained_models` directory:

- `best_model_MyCNN_2d.pth`: 2D CNN model trained on spectrograms
- `best_model_MyCNN_1d.pth`: 1D CNN model trained on raw waveforms
- `best_model_MyResCNN_2d.pth`: Residual CNN model trained on spectrograms
- `best_model_SeismicCNN_2d.pth`: Alternative CNN architecture trained on spectrograms

These models classify seismic events into four categories:
1. Earthquake
2. Explosion
3. Noise
4. Surface event

## Troubleshooting

- **ImportError**: Ensure you have all required packages installed. You need PyTorch, ObsPy, SciPy, and NumPy.
- **GPU Memory Issues**: If you run into GPU memory issues, use `--device cpu` to fall back to CPU processing.
- **Model Loading Errors**: Make sure the model file path is correct and the model architecture matches the saved weights.
- **Missing Components**: The model requires all three components (Z, N, E) for accurate classification.

If you encounter any issues, feel free to open an issue on the repository.
