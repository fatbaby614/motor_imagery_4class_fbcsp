"""
Example prediction script for motor imagery classification.

This script demonstrates how to load a trained model and make predictions
on new EEG data.
"""

import numpy as np
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader


def main():
    print("=" * 60)
    print("Motor Imagery Prediction Example")
    print("=" * 60)
    
    # Load trained model
    print("\n1. Loading trained model...")
    try:
        with open('motor_imagery_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = model_data['classifier']
        preprocessor = model_data['preprocessor']
        config = model_data['config']
        
        print("   Model loaded successfully")
        print(f"   Sample rate: {config['sample_rate']} Hz")
        print(f"   Number of channels: {config['n_channels']}")
        print(f"   Classes: {config['class_labels']}")
    except FileNotFoundError:
        print("   Error: Model file not found. Please run train_example.py first.")
        return
    
    # Generate test data
    print("\n2. Generating test data...")
    X_test, y_test = DataLoader.create_synthetic_data(
        n_trials=20,
        n_channels=config['n_channels'],
        n_samples=int(4.0 * config['sample_rate']),  # 4 seconds
        n_classes=4,
        sample_rate=config['sample_rate']
    )
    
    print(f"   Test data shape: {X_test.shape}")
    
    # Preprocess test data
    print("\n3. Preprocessing test data...")
    X_test_preprocessed = preprocessor.preprocess(X_test)
    print("   Preprocessing complete")
    
    # Make predictions
    print("\n4. Making predictions...")
    predictions = classifier.predict(X_test_preprocessed)
    probabilities = classifier.predict_proba(X_test_preprocessed)
    
    # Display results
    print("\n5. Prediction Results:")
    print("-" * 60)
    
    for i in range(len(X_test)):
        true_label = config['class_labels'][y_test[i]]
        pred_label = config['class_labels'][predictions[i]]
        confidence = probabilities[i, predictions[i]] * 100
        
        correct = "✓" if predictions[i] == y_test[i] else "✗"
        
        print(f"Trial {i+1:2d}: True={true_label:12s} | Pred={pred_label:12s} | "
              f"Confidence={confidence:5.1f}% | {correct}")
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print("-" * 60)
    print(f"Overall Accuracy: {accuracy:.2%} ({np.sum(predictions == y_test)}/{len(y_test)})")
    
    # Show confusion matrix
    print("\n6. Confusion Matrix:")
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, predictions)
    class_names = [config['class_labels'][i] for i in sorted(config['class_labels'].keys())]
    
    # Print header
    print("\n" + " " * 15 + "Predicted")
    print(" " * 12 + "  ".join([f"{name[:4]:>4s}" for name in class_names]))
    
    # Print rows
    for i, true_class in enumerate(class_names):
        row_str = f"{true_class:12s}"
        for j in range(len(class_names)):
            row_str += f"  {cm[i, j]:4d}"
        if i == 0:
            row_str = "True " + row_str
        else:
            row_str = "     " + row_str
        print(row_str)
    
    print("\n" + "=" * 60)
    print("Prediction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
