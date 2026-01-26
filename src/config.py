"""
Configuration settings for motor imagery classification.
"""


class Config:
    """Configuration class for motor imagery FBCSP system."""
    
    # EEG acquisition settings
    SAMPLE_RATE = 250.0  # Hz
    N_CHANNELS = 8  # Number of EEG channels
    
    # Preprocessing settings
    BANDPASS_LOW = 8.0  # Hz - lower cutoff for bandpass filter
    BANDPASS_HIGH = 30.0  # Hz - upper cutoff for bandpass filter
    FILTER_ORDER = 5  # Order of Butterworth filter
    
    # FBCSP settings
    FILTER_BANK = [
        (8, 12),   # mu band
        (12, 16),  # low beta
        (16, 20),  # mid beta
        (20, 24),  # high beta
        (24, 28),  # high beta
        (28, 32),  # high beta
    ]
    N_CSP_COMPONENTS = 2  # CSP components per filter
    
    # Classification settings
    CLASSIFIER_TYPE = 'lda'  # 'lda' or 'svm'
    
    # Motor imagery classes
    CLASS_LABELS = {
        0: 'Left Hand',
        1: 'Right Hand',
        2: 'Feet',
        3: 'Tongue'
    }
    
    # Epoch settings
    EPOCH_START = 0.0  # seconds relative to cue
    EPOCH_END = 4.0  # seconds relative to cue
    BASELINE_START = 0.0  # seconds
    BASELINE_END = 0.5  # seconds
    
    # Cross-validation
    CV_FOLDS = 5
    
    # OpenBCI specific settings
    OPENBCI_CHANNELS = 8
    OPENBCI_SAMPLE_RATE = 250.0
