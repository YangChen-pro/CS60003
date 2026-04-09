"""Basic regression tests for the handcrafted MLP."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp_hw1.metrics import confusion_matrix
from mlp_hw1.model import ThreeLayerMLP


class ThreeLayerMLPTest(unittest.TestCase):
    """Small tests that do not require the EuroSAT dataset."""

    def test_forward_shape(self) -> None:
        model = ThreeLayerMLP(input_dim=6, hidden_dim=8, output_dim=3, activation="relu", xp=np)
        features = np.random.randn(4, 6).astype(np.float32)
        logits = model.forward(features)
        self.assertEqual(logits.shape, (4, 3))

    def test_training_step_reduces_loss(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(48, 4)).astype(np.float32)
        labels = (features[:, 0] + 0.8 * features[:, 1] > 0).astype(np.int64)
        model = ThreeLayerMLP(input_dim=4, hidden_dim=16, output_dim=2, activation="tanh", xp=np, seed=0)
        initial_loss = model.compute_loss(features, labels)
        for _ in range(120):
            model.loss_and_backward(features, labels, weight_decay=0.0)
            model.step(learning_rate=0.05)
        final_loss = model.compute_loss(features, labels)
        self.assertLess(final_loss, initial_loss)

    def test_confusion_matrix_counts(self) -> None:
        matrix = confusion_matrix(
            np.array([0, 0, 1, 2, 2]),
            np.array([0, 1, 1, 2, 0]),
            num_classes=3,
        )
        expected = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        np.testing.assert_array_equal(matrix, expected)


if __name__ == "__main__":
    unittest.main()
