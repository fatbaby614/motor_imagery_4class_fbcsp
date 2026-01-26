"""
Data loading utilities for motor imagery EEG data.

Supports loading data from various formats including OpenBCI and standard BCI datasets.
"""

import numpy as np
import os


class DataLoader:
    """
    Utility class for loading motor imagery EEG data.
    
    Supports multiple data formats and provides a unified interface.
    """
    
    @staticmethod
    def load_openbci_csv(filepath, n_channels=8):
        """
        Load data from OpenBCI CSV format.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file
        n_channels : int, default=8
            Number of EEG channels
            
        Returns
        -------
        data : ndarray, shape (n_channels, n_samples)
            EEG data
        timestamps : ndarray, shape (n_samples,)
            Sample timestamps
        """
        import pandas as pd
        
        df = pd.read_csv(filepath)
        
        # OpenBCI CSV typically has columns: Sample Index, EEG Channel 1, ..., EEG Channel N, ...
        # Extract EEG channels (usually the first n_channels columns after index)
        channel_cols = [col for col in df.columns if 'EEG' in col or 'Channel' in col][:n_channels]
        
        data = df[channel_cols].values.T
        
        # Try to get timestamps
        if 'Time' in df.columns or 'Timestamp' in df.columns:
            time_col = 'Time' if 'Time' in df.columns else 'Timestamp'
            timestamps = df[time_col].values
        else:
            timestamps = np.arange(data.shape[1])
        
        return data, timestamps
    
    @staticmethod
    def load_numpy(filepath):
        """
        Load data from NumPy format.
        
        Parameters
        ----------
        filepath : str
            Path to .npy or .npz file
            
        Returns
        -------
        data : dict or ndarray
            Loaded data
        """
        if filepath.endswith('.npz'):
            return dict(np.load(filepath))
        else:
            return np.load(filepath)
    
    @staticmethod
    def save_numpy(filepath, data, **kwargs):
        """
        Save data to NumPy format.
        
        Parameters
        ----------
        filepath : str
            Path to save file
        data : ndarray
            Data to save
        **kwargs : dict
            Additional arrays to save (for .npz format)
        """
        if kwargs:
            np.savez(filepath, data=data, **kwargs)
        else:
            np.save(filepath, data)
    
    @staticmethod
    def create_synthetic_data(n_trials=100, n_channels=8, n_samples=1000, n_classes=4, sample_rate=250.0):
        """
        Create synthetic motor imagery data for testing.
        
        Parameters
        ----------
        n_trials : int, default=100
            Number of trials
        n_channels : int, default=8
            Number of EEG channels
        n_samples : int, default=1000
            Number of samples per trial
        n_classes : int, default=4
            Number of classes
        sample_rate : float, default=250.0
            Sampling rate in Hz
            
        Returns
        -------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            Synthetic EEG data
        y : ndarray, shape (n_trials,)
            Class labels
        """
        X = np.zeros((n_trials, n_channels, n_samples))
        y = np.zeros(n_trials, dtype=int)
        
        # Generate synthetic data with class-specific frequency content
        t = np.arange(n_samples) / sample_rate
        
        for i in range(n_trials):
            class_label = i % n_classes
            y[i] = class_label
            
            # Different frequency emphasis for each class
            if class_label == 0:
                # Class 0: Left hand - emphasis on mu rhythm (10 Hz) in C3
                freq = 10
                for ch in range(n_channels):
                    amplitude = 2.0 if ch in [2, 3] else 1.0
                    X[i, ch, :] = amplitude * np.sin(2 * np.pi * freq * t)
            elif class_label == 1:
                # Class 1: Right hand - emphasis on mu rhythm (10 Hz) in C4
                freq = 10
                for ch in range(n_channels):
                    amplitude = 2.0 if ch in [4, 5] else 1.0
                    X[i, ch, :] = amplitude * np.sin(2 * np.pi * freq * t)
            elif class_label == 2:
                # Class 2: Feet - emphasis on beta rhythm (20 Hz) in Cz
                freq = 20
                for ch in range(n_channels):
                    amplitude = 2.0 if ch in [0, 1] else 1.0
                    X[i, ch, :] = amplitude * np.sin(2 * np.pi * freq * t)
            else:
                # Class 3: Tongue - emphasis on higher beta (25 Hz)
                freq = 25
                for ch in range(n_channels):
                    amplitude = 1.5
                    X[i, ch, :] = amplitude * np.sin(2 * np.pi * freq * t)
            
            # Add noise
            X[i] += np.random.randn(n_channels, n_samples) * 0.5
        
        return X, y


def load_bci_competition_data(dataset_path):
    """
    Load data from BCI Competition format.
    
    This is a placeholder for loading standard BCI Competition datasets.
    Specific implementation depends on the dataset format.
    
    Parameters
    ----------
    dataset_path : str
        Path to dataset directory
        
    Returns
    -------
    X : ndarray
        EEG data
    y : ndarray
        Labels
    """
    # This would need to be implemented based on specific dataset format
    raise NotImplementedError("BCI Competition data loading needs to be implemented for specific datasets")
