"""
Classification module for motor imagery EEG data.

Provides classifiers and utilities for 4-class motor imagery classification
using FBCSP features.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from .fbcsp import FBCSP
except ImportError:
    from fbcsp import FBCSP


class MotorImageryClassifier:
    """
    Four-class motor imagery classifier using FBCSP and one-vs-rest strategy.
    
    This classifier handles 4-class motor imagery classification by training
    binary FBCSP classifiers in a one-vs-rest manner.
    
    Parameters
    ----------
    classifier_type : str, default='lda'
        Type of classifier to use ('lda' or 'svm')
    filter_bank : list of tuple, default=None
        Filter bank specification for FBCSP
    n_components : int, default=2
        Number of CSP components per filter
    sample_rate : float, default=250.0
        Sampling rate in Hz
    """
    
    def __init__(self, classifier_type='lda', filter_bank=None, n_components=2, sample_rate=250.0):
        self.classifier_type = classifier_type
        self.filter_bank = filter_bank
        self.n_components = n_components
        self.sample_rate = sample_rate
        
        # One-vs-rest classifiers
        self.classifiers = {}
        self.fbcsp_models = {}
        self.classes_ = None
        
    def _create_classifier(self):
        """Create a base classifier."""
        if self.classifier_type == 'lda':
            return LDA()
        elif self.classifier_type == 'svm':
            return SVC(kernel='rbf', gamma='scale', probability=True)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def fit(self, X, y):
        """
        Fit the 4-class motor imagery classifier.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        y : ndarray, shape (n_trials,)
            Labels (0, 1, 2, 3 for four classes)
            
        Returns
        -------
        self : object
            Returns self
        """
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 4:
            raise ValueError("This classifier is designed for 4-class problems")
        
        # Train one-vs-rest binary classifiers for each class
        for target_class in self.classes_:
            # Create binary labels (current class vs rest)
            y_binary = (y == target_class).astype(int)
            
            # Create and fit FBCSP
            fbcsp = FBCSP(
                filter_bank=self.filter_bank,
                n_components=self.n_components,
                sample_rate=self.sample_rate
            )
            
            # Fit FBCSP
            fbcsp.fit(X, y_binary)
            
            # Extract features
            features = fbcsp.transform(X)
            
            # Create and train classifier
            classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', self._create_classifier())
            ])
            classifier.fit(features, y_binary)
            
            # Store models
            self.fbcsp_models[target_class] = fbcsp
            self.classifiers[target_class] = classifier
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
            
        Returns
        -------
        predictions : ndarray, shape (n_trials,)
            Predicted class labels
        """
        # Get probability scores from each one-vs-rest classifier
        all_scores = []
        
        for target_class in self.classes_:
            # Extract features using the corresponding FBCSP
            features = self.fbcsp_models[target_class].transform(X)
            
            # Get decision scores
            if hasattr(self.classifiers[target_class].named_steps['classifier'], 'decision_function'):
                scores = self.classifiers[target_class].decision_function(features)
            else:
                # For probability-based classifiers
                proba = self.classifiers[target_class].predict_proba(features)
                scores = proba[:, 1]  # Probability of positive class
            
            all_scores.append(scores)
        
        # Stack scores and find class with highest score
        all_scores = np.column_stack(all_scores)
        predictions = self.classes_[np.argmax(all_scores, axis=1)]
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
            
        Returns
        -------
        probabilities : ndarray, shape (n_trials, n_classes)
            Class probabilities
        """
        # Get scores from each one-vs-rest classifier
        all_scores = []
        
        for target_class in self.classes_:
            features = self.fbcsp_models[target_class].transform(X)
            
            if hasattr(self.classifiers[target_class].named_steps['classifier'], 'predict_proba'):
                proba = self.classifiers[target_class].predict_proba(features)
                scores = proba[:, 1]
            else:
                scores = self.classifiers[target_class].decision_function(features)
            
            all_scores.append(scores)
        
        # Normalize scores to probabilities
        all_scores = np.column_stack(all_scores)
        # Apply softmax
        exp_scores = np.exp(all_scores - np.max(all_scores, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            Test EEG data
        y : ndarray, shape (n_trials,)
            True labels
            
        Returns
        -------
        score : float
            Mean accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        y : ndarray, shape (n_trials,)
            Labels
        cv : int, default=5
            Number of cross-validation folds
            
        Returns
        -------
        scores : dict
            Dictionary containing mean and std of CV scores
        """
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create a new classifier for each fold
            clf = MotorImageryClassifier(
                classifier_type=self.classifier_type,
                filter_bank=self.filter_bank,
                n_components=self.n_components,
                sample_rate=self.sample_rate
            )
            
            # Fit and score
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            cv_scores.append(score)
        
        return {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'all_scores': cv_scores
        }
