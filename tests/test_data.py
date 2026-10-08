"""Tests for the data loading module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from PIL import Image
from xray_classifier.data import get_dataloaders, get_transforms


def make_fake_split(root, per_class=20):
    rng = np.random.default_rng(0)
    for split in ["train", "test"]:
        for cls in ["A", "B"]:
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(per_class):
                array = rng.integers(0, 256, (16, 16), dtype=np.uint8)
                Image.fromarray(array).save(folder / f"{i}.png")


def test_get_transforms_returns_composable_pipeline():
    transform = get_transforms()
    assert transform is not None
    # resize, grayscale, to_tensor, normalize
    assert len(transform.transforms) == 4


def test_validation_split_is_disjoint_and_stratified(tmp_path):
    make_fake_split(tmp_path)
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        str(tmp_path / "train"), str(tmp_path / "test"), batch_size=4, val_fraction=0.25
    )
    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)
    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == 40
    assert len(val_idx) == 10
    targets = train_loader.dataset.dataset.targets
    assert sum(1 for i in val_idx if targets[i] == 0) == 5
    assert len(test_loader.dataset) == 40
    assert classes == ["A", "B"]
