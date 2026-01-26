"""
Example training script for 4-class motor imagery classification.

This script demonstrates how to train a motor imagery classifier using
the FBCSP algorithm with synthetic data.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier import MotorImageryClassifier
from src.preprocessing import Preprocessor
from src.data_loader import DataLoader
from src.config import Config


def main():
    print("=" * 60)
    print("Motor Imagery 4-Class FBCSP Training Example")
    print("=" * 60)
    
    # Create synthetic data for demonstration
    print("\n1. Generating synthetic motor imagery data...")
    n_trials = 200
    n_channels = Config.N_CHANNELS
    n_samples = int((Config.EPOCH_END - Config.EPOCH_START) * Config.SAMPLE_RATE)
    
    X, y = DataLoader.create_synthetic_data(
        n_trials=n_trials,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=4,
        sample_rate=Config.SAMPLE_RATE
    )
    
    print(f"   Data shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = Preprocessor(
        sample_rate=Config.SAMPLE_RATE,
        low_freq=Config.BANDPASS_LOW,
        high_freq=Config.BANDPASS_HIGH,
        filter_order=Config.FILTER_ORDER
    )
    
    X_preprocessed = preprocessor.preprocess(X)
    print(f"   Preprocessing complete")
    
    # Split data into train and test
    print("\n3. Splitting data into train and test sets...")
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} trials")
    print(f"   Test set: {X_test.shape[0]} trials")
    
    # Create and train classifier
    print("\n4. Training FBCSP classifier...")
    classifier = MotorImageryClassifier(
        classifier_type=Config.CLASSIFIER_TYPE,
        filter_bank=Config.FILTER_BANK,
        n_components=Config.N_CSP_COMPONENTS,
        sample_rate=Config.SAMPLE_RATE
    )
    
    classifier.fit(X_train, y_train)
    print("   Training complete")
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_accuracy = classifier.score(X_test, y_test)
    print(f"   Test Accuracy: {test_accuracy:.2%}")
    
    # Predict on test samples
    predictions = classifier.predict(X_test[:10])
    probabilities = classifier.predict_proba(X_test[:10])
    
    print("\n6. Sample predictions:")
    for i in range(min(10, len(X_test))):
        true_label = Config.CLASS_LABELS[y_test[i]]
        pred_label = Config.CLASS_LABELS[predictions[i]]
        confidence = probabilities[i, predictions[i]] * 100
        print(f"   Trial {i+1}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.1f}%")
    
    # Perform cross-validation
    print("\n7. Performing cross-validation...")
    cv_results = classifier.cross_validate(X_preprocessed, y, cv=Config.CV_FOLDS)
    print(f"   Mean CV Accuracy: {cv_results['mean_accuracy']:.2%} ± {cv_results['std_accuracy']:.2%}")
    print(f"   Fold scores: {[f'{s:.2%}' for s in cv_results['all_scores']]}")
    
    # Save model (optional)
    print("\n8. Saving model...")
    import pickle
    
    model_data = {
        'classifier': classifier,
        'preprocessor': preprocessor,
        'config': {
            'sample_rate': Config.SAMPLE_RATE,
            'n_channels': Config.N_CHANNELS,
            'class_labels': Config.CLASS_LABELS
        }
    }
    
    with open('motor_imagery_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("   Model saved to 'motor_imagery_model.pkl'")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
