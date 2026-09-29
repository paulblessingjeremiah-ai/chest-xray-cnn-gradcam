"""Tests for the model module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from xray_classifier.model import build_model


def test_build_model_output_shape():
    model = build_model(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 4)


def test_build_model_different_num_classes():
    model = build_model(num_classes=2)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 2)
