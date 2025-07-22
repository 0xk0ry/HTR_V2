"""
HTR Model Package

This package contains the implementation of a Handwritten Text Recognition (HTR) model
using CvT (Convolutional Vision Transformer) as the backbone with chunking strategy.
"""

from .HTR_ME import (
    HTRModel,
    CTCDecoder,
    ImageChunker,
    CvT,
    train_epoch,
    validate,
    inference_example,
    create_model_example
)

__version__ = "1.0.0"
__author__ = "HTR Team"

__all__ = [
    "HTRModel",
    "CTCDecoder", 
    "ImageChunker",
    "CvT",
    "train_epoch",
    "validate",
    "inference_example",
    "create_model_example"
]
