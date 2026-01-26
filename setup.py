from setuptools import setup, find_packages

setup(
    name='motor_imagery_4class_fbcsp',
    version='0.1.0',
    description='Four-class motor imagery classification using Filter Bank Common Spatial Patterns for OpenBCI',
    author='fatbaby614',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.8.0',
        'scikit-learn>=1.0.1',
        'mne>=1.0.0',
        'matplotlib>=3.4.0',
        'pandas>=1.3.0',
    ],
    python_requires='>=3.7',
)
