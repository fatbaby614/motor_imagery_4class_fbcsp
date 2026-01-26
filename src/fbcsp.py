"""
Filter Bank Common Spatial Patterns (FBCSP) implementation for motor imagery classification.

This module implements the FBCSP algorithm which combines filter banks with CSP
for improved motor imagery classification performance.
"""

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns (CSP) implementation.
    
    CSP is a supervised decomposition method for extracting spatial filters
    that maximize the variance of one class while minimizing the variance
    of the other class.
    
    Parameters
    ----------
    n_components : int, default=4
        Number of CSP components to extract (filters from each end of the spectrum)
    """
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
        self.eigenvalues_ = None
        
    def fit(self, X, y):
        """
        Fit CSP filters.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        y : ndarray, shape (n_trials,)
            Labels (0 or 1 for binary classification)
            
        Returns
        -------
        self : object
            Returns self
        """
        # Get unique classes
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes for binary classification")
        
        # Calculate covariance matrices for each class
        cov_1 = self._compute_covariance(X[y == classes[0]])
        cov_2 = self._compute_covariance(X[y == classes[1]])
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(cov_1, cov_1 + cov_2)
        
        # Sort by eigenvalues
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        
        # Select filters (n_components from each end)
        n_comp = self.n_components
        selected_indices = np.concatenate([
            np.arange(n_comp),
            np.arange(len(eigenvalues) - n_comp, len(eigenvalues))
        ])
        
        self.filters_ = eigenvectors[:, selected_indices].T
        self.eigenvalues_ = eigenvalues[selected_indices]
        
        return self
    
    def transform(self, X):
        """
        Apply CSP filters to extract features.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
            
        Returns
        -------
        features : ndarray, shape (n_trials, n_components * 2)
            CSP features (log variance)
        """
        if self.filters_ is None:
            raise ValueError("CSP must be fitted before transform")
        
        n_trials = X.shape[0]
        n_filters = self.filters_.shape[0]
        features = np.zeros((n_trials, n_filters))
        
        for i in range(n_trials):
            # Apply spatial filters
            filtered = np.dot(self.filters_, X[i])
            
            # Compute log variance as features
            variance = np.var(filtered, axis=1)
            features[i] = np.log(variance / np.sum(variance))
        
        return features
    
    def _compute_covariance(self, X):
        """
        Compute average covariance matrix.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data for one class
            
        Returns
        -------
        cov : ndarray, shape (n_channels, n_channels)
            Average covariance matrix
        """
        n_trials, n_channels, n_samples = X.shape
        cov = np.zeros((n_channels, n_channels))
        
        for trial in X:
            # Normalize trial
            trial = trial - np.mean(trial, axis=1, keepdims=True)
            # Compute covariance
            cov_trial = np.dot(trial, trial.T) / n_samples
            # Normalize by trace
            cov += cov_trial / np.trace(cov_trial)
        
        # Average over trials
        cov /= n_trials
        
        return cov


class FBCSP(BaseEstimator, TransformerMixin):
    """
    Filter Bank Common Spatial Patterns (FBCSP).
    
    FBCSP applies multiple bandpass filters to create a filter bank, then
    applies CSP to each filtered signal. This allows capturing discriminative
    information from different frequency bands.
    
    Parameters
    ----------
    filter_bank : list of tuple, default=None
        List of (low_freq, high_freq) tuples defining the filter bank.
        If None, uses default filter bank: [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
    n_components : int, default=2
        Number of CSP components per filter
    sample_rate : float, default=250.0
        Sampling rate in Hz
    filter_order : int, default=5
        Order of Butterworth bandpass filter
    """
    
    def __init__(self, filter_bank=None, n_components=2, sample_rate=250.0, filter_order=5):
        if filter_bank is None:
            # Default filter bank covering mu (8-12 Hz) and beta (12-30 Hz) bands
            self.filter_bank = [
                (8, 12),   # mu band
                (12, 16),  # low beta
                (16, 20),  # mid beta
                (20, 24),  # high beta
                (24, 28),  # high beta
                (28, 32),  # high beta
            ]
        else:
            self.filter_bank = filter_bank
        
        self.n_components = n_components
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.csp_list = []
        
    def fit(self, X, y):
        """
        Fit FBCSP.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        y : ndarray, shape (n_trials,)
            Labels for binary classification
            
        Returns
        -------
        self : object
            Returns self
        """
        self.csp_list = []
        
        # For each frequency band in filter bank
        for low_freq, high_freq in self.filter_bank:
            # Filter the data
            X_filtered = self._apply_bandpass(X, low_freq, high_freq)
            
            # Fit CSP for this band
            csp = CSP(n_components=self.n_components)
            csp.fit(X_filtered, y)
            
            self.csp_list.append(csp)
        
        return self
    
    def transform(self, X):
        """
        Transform data using FBCSP.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
            
        Returns
        -------
        features : ndarray, shape (n_trials, n_bands * n_components * 2)
            FBCSP features
        """
        if not self.csp_list:
            raise ValueError("FBCSP must be fitted before transform")
        
        all_features = []
        
        # For each frequency band
        for i, (low_freq, high_freq) in enumerate(self.filter_bank):
            # Filter the data
            X_filtered = self._apply_bandpass(X, low_freq, high_freq)
            
            # Extract CSP features
            features = self.csp_list[i].transform(X_filtered)
            all_features.append(features)
        
        # Concatenate features from all bands
        return np.concatenate(all_features, axis=1)
    
    def _apply_bandpass(self, X, low_freq, high_freq):
        """
        Apply bandpass filter to data.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        low_freq : float
            Low cutoff frequency in Hz
        high_freq : float
            High cutoff frequency in Hz
            
        Returns
        -------
        X_filtered : ndarray, shape (n_trials, n_channels, n_samples)
            Filtered EEG data
        """
        nyquist = self.sample_rate / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(self.filter_order, [low, high], btype='band')
        
        # Apply filter to each trial and channel
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_filtered[i, j, :] = signal.filtfilt(b, a, X[i, j, :])
        
        return X_filtered
