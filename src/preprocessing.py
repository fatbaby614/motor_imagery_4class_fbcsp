"""
Preprocessing module for EEG motor imagery data.

Provides utilities for loading, filtering, and preparing EEG data
for motor imagery classification.
"""

import numpy as np
from scipy import signal


class Preprocessor:
    """
    EEG data preprocessor for motor imagery tasks.
    
    Parameters
    ----------
    sample_rate : float, default=250.0
        Sampling rate in Hz
    low_freq : float, default=8.0
        Low cutoff frequency for bandpass filter
    high_freq : float, default=30.0
        High cutoff frequency for bandpass filter
    filter_order : int, default=5
        Order of Butterworth filter
    """
    
    def __init__(self, sample_rate=250.0, low_freq=8.0, high_freq=30.0, filter_order=5):
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.filter_order = filter_order
        
    def bandpass_filter(self, data):
        """
        Apply bandpass filter to EEG data.
        
        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples) or (n_trials, n_channels, n_samples)
            EEG data to filter
            
        Returns
        -------
        filtered_data : ndarray
            Bandpass filtered data (same shape as input)
        """
        nyquist = self.sample_rate / 2.0
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(self.filter_order, [low, high], btype='band')
        
        # Handle both 2D and 3D arrays
        if data.ndim == 2:
            # Single trial: (n_channels, n_samples)
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
        elif data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_samples)
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    filtered_data[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
        else:
            raise ValueError("Data must be 2D or 3D array")
        
        return filtered_data
    
    def epoch_data(self, data, events, epoch_start, epoch_end):
        """
        Extract epochs from continuous data.
        
        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Continuous EEG data
        events : ndarray, shape (n_events, 2)
            Event array where each row is [sample_index, event_type]
        epoch_start : float
            Start time of epoch relative to event (in seconds)
        epoch_end : float
            End time of epoch relative to event (in seconds)
            
        Returns
        -------
        epochs : ndarray, shape (n_valid_events, n_channels, n_epoch_samples)
            Epoched data
        labels : ndarray, shape (n_valid_events,)
            Event types
            
        Notes
        -----
        Epochs that fall outside the data range are skipped. The returned arrays
        may have fewer trials than the input events array.
        """
        start_sample = int(epoch_start * self.sample_rate)
        end_sample = int(epoch_end * self.sample_rate)
        epoch_length = end_sample - start_sample
        
        n_channels = data.shape[0]
        
        valid_epochs = []
        valid_labels = []
        
        for i, (sample_idx, event_type) in enumerate(events):
            sample_idx = int(sample_idx)
            epoch_start_idx = sample_idx + start_sample
            epoch_end_idx = sample_idx + end_sample
            
            # Skip if epoch is out of bounds
            if epoch_start_idx < 0 or epoch_end_idx > data.shape[1]:
                continue
            
            epoch = data[:, epoch_start_idx:epoch_end_idx]
            valid_epochs.append(epoch)
            valid_labels.append(int(event_type))
        
        if len(valid_epochs) == 0:
            raise ValueError("No valid epochs found. Check event timings and data length.")
        
        epochs = np.array(valid_epochs)
        labels = np.array(valid_labels, dtype=int)
        
        return epochs, labels
    
    def remove_baseline(self, epochs, baseline_start=0, baseline_end=0.5):
        """
        Remove baseline from epochs.
        
        Parameters
        ----------
        epochs : ndarray, shape (n_trials, n_channels, n_samples)
            Epoched EEG data
        baseline_start : float, default=0
            Start time of baseline period (in seconds from epoch start)
        baseline_end : float, default=0.5
            End time of baseline period (in seconds from epoch start)
            
        Returns
        -------
        corrected_epochs : ndarray
            Baseline-corrected epochs
        """
        start_idx = int(baseline_start * self.sample_rate)
        end_idx = int(baseline_end * self.sample_rate)
        
        corrected_epochs = epochs.copy()
        
        for i in range(epochs.shape[0]):
            baseline = np.mean(epochs[i, :, start_idx:end_idx], axis=1, keepdims=True)
            corrected_epochs[i] = epochs[i] - baseline
        
        return corrected_epochs
    
    def normalize_data(self, data):
        """
        Normalize data by standardizing each channel.
        
        Parameters
        ----------
        data : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
            
        Returns
        -------
        normalized_data : ndarray
            Normalized data (zero mean, unit variance per channel)
        """
        normalized_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                channel_data = data[i, j, :]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                if std > 0:
                    normalized_data[i, j, :] = (channel_data - mean) / std
                else:
                    normalized_data[i, j, :] = channel_data - mean
        
        return normalized_data
    
    def preprocess(self, data, apply_bandpass=True, apply_normalization=True):
        """
        Apply full preprocessing pipeline.
        
        Parameters
        ----------
        data : ndarray
            EEG data to preprocess
        apply_bandpass : bool, default=True
            Whether to apply bandpass filter
        apply_normalization : bool, default=True
            Whether to normalize data
            
        Returns
        -------
        processed_data : ndarray
            Preprocessed data
        """
        processed_data = data.copy()
        
        if apply_bandpass:
            processed_data = self.bandpass_filter(processed_data)
        
        if apply_normalization:
            processed_data = self.normalize_data(processed_data)
        
        return processed_data
