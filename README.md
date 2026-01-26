# Motor Imagery 4-Class FBCSP

OpenBCI四分类运动想象 (OpenBCI Four-Class Motor Imagery Classification)

A Python implementation of Filter Bank Common Spatial Patterns (FBCSP) for classifying four-class motor imagery brain signals from OpenBCI devices.

## Overview

This project implements a complete pipeline for motor imagery EEG classification using the FBCSP (Filter Bank Common Spatial Patterns) algorithm. It supports classification of four motor imagery tasks:

- **Class 0**: Left Hand movement imagery
- **Class 1**: Right Hand movement imagery  
- **Class 2**: Feet movement imagery
- **Class 3**: Tongue movement imagery

## Features

- ✅ Filter Bank Common Spatial Patterns (FBCSP) implementation
- ✅ One-vs-rest classification strategy for 4-class problems
- ✅ Comprehensive EEG preprocessing pipeline
- ✅ Support for OpenBCI data format
- ✅ LDA and SVM classifiers
- ✅ Cross-validation support
- ✅ Example scripts for training and prediction

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- MNE (for EEG processing)
- matplotlib
- pandas

### Install from source

```bash
# Clone the repository
git clone https://github.com/fatbaby614/motor_imagery_4class_fbcsp.git
cd motor_imagery_4class_fbcsp

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Training a Classifier

```python
from src.classifier import MotorImageryClassifier
from src.preprocessing import Preprocessor
from src.data_loader import DataLoader
from src.config import Config

# Load your EEG data
# X shape: (n_trials, n_channels, n_samples)
# y shape: (n_trials,) with values 0, 1, 2, 3

# Create preprocessor
preprocessor = Preprocessor(
    sample_rate=Config.SAMPLE_RATE,
    low_freq=Config.BANDPASS_LOW,
    high_freq=Config.BANDPASS_HIGH
)

# Preprocess data
X_preprocessed = preprocessor.preprocess(X)

# Create and train classifier
classifier = MotorImageryClassifier(
    classifier_type='lda',
    filter_bank=Config.FILTER_BANK,
    n_components=2,
    sample_rate=Config.SAMPLE_RATE
)

classifier.fit(X_preprocessed, y)

# Make predictions
predictions = classifier.predict(X_test)
accuracy = classifier.score(X_test, y_test)
```

### Running Examples

Train a model with synthetic data:

```bash
cd examples
python train_example.py
```

Make predictions with a trained model:

```bash
cd examples
python predict_example.py
```

## Project Structure

```
motor_imagery_4class_fbcsp/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── fbcsp.py              # FBCSP and CSP implementation
│   ├── preprocessing.py      # EEG preprocessing utilities
│   ├── classifier.py         # 4-class classifier implementation
│   ├── data_loader.py        # Data loading utilities
│   └── config.py             # Configuration settings
├── examples/
│   ├── train_example.py      # Training example script
│   └── predict_example.py    # Prediction example script
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## Algorithm Details

### FBCSP (Filter Bank Common Spatial Patterns)

FBCSP is an extension of the CSP algorithm that uses multiple frequency bands to extract more discriminative features:

1. **Filter Bank**: The EEG signal is filtered into multiple frequency bands (typically covering mu and beta rhythms: 8-32 Hz)
2. **CSP per Band**: CSP is applied to each filtered signal to extract spatial patterns
3. **Feature Extraction**: Log-variance features are computed from the filtered signals
4. **Classification**: Features from all bands are concatenated and classified

### One-vs-Rest Strategy

For 4-class classification, we use a one-vs-rest approach:
- Train 4 binary classifiers, each distinguishing one class from the others
- For prediction, select the class with the highest confidence score

## Configuration

Key parameters can be configured in `src/config.py`:

- **Sample Rate**: Default 250 Hz (OpenBCI standard)
- **Filter Bank**: Default covers 8-32 Hz in 6 bands
- **CSP Components**: Default 2 components per filter
- **Classifier**: LDA (Linear Discriminant Analysis) or SVM

## Data Format

### Input Data Format

EEG data should be in the following format:

- **X**: NumPy array of shape `(n_trials, n_channels, n_samples)`
  - `n_trials`: Number of trials/epochs
  - `n_channels`: Number of EEG channels (default: 8 for OpenBCI)
  - `n_samples`: Number of time samples per trial
  
- **y**: NumPy array of shape `(n_trials,)` with integer labels 0, 1, 2, 3

### OpenBCI Data

For loading OpenBCI CSV files:

```python
from src.data_loader import DataLoader

data, timestamps = DataLoader.load_openbci_csv('path/to/data.csv', n_channels=8)
```

## Performance

With proper data and preprocessing, the FBCSP algorithm typically achieves:
- **Binary classification**: 70-90% accuracy
- **4-class classification**: 50-70% accuracy

Performance depends on:
- Data quality and quantity
- Subject-specific variability
- Proper hyperparameter tuning
- Training data quality

## References

1. Ang, K. K., Chin, Z. Y., Zhang, H., & Guan, C. (2008). Filter bank common spatial pattern (FBCSP) in brain-computer interface. *IEEE International Joint Conference on Neural Networks*.

2. Ramoser, H., Muller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering*.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{motor_imagery_fbcsp,
  title={Motor Imagery 4-Class FBCSP},
  author={fatbaby614},
  year={2026},
  url={https://github.com/fatbaby614/motor_imagery_4class_fbcsp}
}
```

## Contact

For questions or issues, please open an issue on GitHub.