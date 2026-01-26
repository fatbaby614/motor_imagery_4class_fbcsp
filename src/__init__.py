# Motor Imagery 4-Class FBCSP Package
__version__ = '0.1.0'

from .fbcsp import FBCSP
from .preprocessing import Preprocessor
from .classifier import MotorImageryClassifier

__all__ = ['FBCSP', 'Preprocessor', 'MotorImageryClassifier']
