# QuakeXNet Seismic Event Classification

This directory contains scripts for applying the QuakeXNet model from SeisBench to classify seismic waveforms.

## Scripts

### 1. `apply_quakexnet.py`

A standalone script that loads a trained QuakeXNet model and applies it to seismic data from FDSN web services.

**Usage:**
```bash
python apply_quakexnet.py --model trained_models/best_model_QuakeXNet_2d.pth \
                         --network UW --station RCM --channel BH \
                         --starttime 2021-01-01T12:00:00 --duration 100 --plot
```

**Arguments:**
- `--model`: Path to the trained model weights
- `--network`: Network code (e.g., UW)
- `--station`: Station code (e.g., RCM)
- `--channel`: Channel code prefix (e.g., BH for BHZ,BHN,BHE)
- `--starttime`: Start time in ISO format (e.g., 2021-01-01T00:00:00)
- `--duration`: Duration in seconds (default: 100.0)
- `--plot`: Plot waveforms and spectrograms
- `--device`: Device to use, 'cpu' or 'cuda' (default: 'cpu')

### 2. `batch_process_waveforms.py`

Batch processes multiple waveform files using the QuakeXNet model.

**Usage:**
```bash
python batch_process_waveforms.py --model trained_models/best_model_QuakeXNet_2d.pth \
                                 --input_dir ../data/binary_classification_waveforms \
                                 --output results_quakexnet.csv \
                                 --max_files 100
```

**Arguments:**
- `--model`: Path to the trained model weights
- `--input_dir`: Directory containing waveform files (default: 'data/binary_classification_waveforms')
- `--output`: Output CSV file for results (default: 'results_quakexnet.csv')
- `--max_files`: Maximum number of files to process (default: 100)
- `--device`: Device to use, 'cpu' or 'cuda' (default: 'cpu')

## SeisBench Integration

The `classifier.py` file contains the QuakeXNet model implementation that inherits from SeisBench's `WaveformModel` class. This allows for full integration with the SeisBench framework.

For direct SeisBench integration:

```python
from classifier import QuakeXNet
import torch
from obspy import Stream, read

# Initialize the model
model = QuakeXNet(
    sampling_rate=50,
    classes=4,
    labels=["eq", "px", "no", "su"],
    num_channels=3,
    num_classes=4,
    dropout_rate=0.4
)

# Load pre-trained weights
model_path = "trained_models/best_model_MyCNN_2d.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Process a stream with SeisBench
stream = read('path/to/mseed/file')
annotations = model.annotate(stream, sampling_rate=50)
detection_results = model.classify_aggregate(annotations)
print(f'Detected events: {detection_results}')
```

## Class Labels

The model classifies seismic events into four categories:
1. Earthquake (eq)
2. Explosion/quarry blast (px)
3. Noise (no)
4. Surface event (su)

## Model Architecture

QuakeXNet is a convolutional neural network architecture designed for seismic event classification. It can process either raw waveforms (1D) or spectrograms (2D). The script uses the 2D version by default, where seismic data is converted to spectrograms before being passed to the model.

## Requirements

- Python 3.7+
- PyTorch
- ObsPy
- NumPy
- SciPy
- Matplotlib
- Pandas
- SeisBench

## Trained Models

Trained model weights are located in the `trained_models` directory. The default model used is `best_model_QuakeXNet_2d.pth`.
