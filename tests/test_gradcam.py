"""Tests for the Grad-CAM module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from xray_classifier.model import build_model
from xray_classifier.gradcam import GradCAM


def test_gradcam_output_shape():
    model = build_model(num_classes=4)
    grad_cam = GradCAM(model, model.layer4[-1])

    dummy_input = torch.randn(1, 3, 224, 224)
    cam, class_idx = grad_cam.generate(dummy_input)

    assert cam.shape == (224, 224)
    assert 0 <= class_idx < 4


def test_gradcam_output_range():
    model = build_model(num_classes=4)
    grad_cam = GradCAM(model, model.layer4[-1])

    dummy_input = torch.randn(1, 3, 224, 224)
    cam, _ = grad_cam.generate(dummy_input)

    assert cam.min() >= 0.0
    assert cam.max() <= 1.0
