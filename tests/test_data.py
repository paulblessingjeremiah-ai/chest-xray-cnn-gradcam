"""Tests for the data loading module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xray_classifier.data import get_transforms


def test_get_transforms_returns_composable_pipeline():
    transform = get_transforms()
    assert transform is not None
    # Should have the expected number of transform steps: resize, grayscale, to_tensor, normalize
    assert len(transform.transforms) == 4
