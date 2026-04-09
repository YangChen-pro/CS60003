"""Three-layer MLP with hand-written backward pass."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backend import to_numpy


@dataclass
class LayerGrads:
    """Gradient buffers for a linear layer."""

    weight: Any
    bias: Any


class ThreeLayerMLP:
    """A simple MLP: input -> hidden -> hidden -> output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_dim2: int,
        output_dim: int,
        activation: str,
        xp: Any,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.activation = activation
        self.xp = xp
        self.dtype = np.float64 if xp is np else xp.float32
        rng = np.random.default_rng(seed)

        # 这里的初始化按激活函数选择，避免一开始梯度过大或过小。
        scale1 = np.sqrt(2.0 / input_dim) if activation == "relu" else np.sqrt(1.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim) if activation == "relu" else np.sqrt(1.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / hidden_dim2) if activation == "relu" else np.sqrt(1.0 / hidden_dim2)
        self.w1 = xp.asarray(rng.standard_normal((input_dim, hidden_dim)) * scale1, dtype=self.dtype)
        self.b1 = xp.zeros(hidden_dim, dtype=self.dtype)
        self.w2 = xp.asarray(rng.standard_normal((hidden_dim, hidden_dim2)) * scale2, dtype=self.dtype)
        self.b2 = xp.zeros(hidden_dim2, dtype=self.dtype)
        self.w3 = xp.asarray(rng.standard_normal((hidden_dim2, output_dim)) * scale3, dtype=self.dtype)
        self.b3 = xp.zeros(output_dim, dtype=self.dtype)
        self.grads = {
            "w1": xp.zeros_like(self.w1),
            "b1": xp.zeros_like(self.b1),
            "w2": xp.zeros_like(self.w2),
            "b2": xp.zeros_like(self.b2),
            "w3": xp.zeros_like(self.w3),
            "b3": xp.zeros_like(self.b3),
        }
        self.cache: dict[str, Any] = {}

    def forward(self, inputs: Any) -> Any:
        """Run the forward pass and cache intermediates."""
        inputs = inputs.astype(self.dtype, copy=False)
        z1 = self._matmul(inputs, self.w1) + self.b1
        h1 = self._activate(z1)
        z2 = self._matmul(h1, self.w2) + self.b2
        h2 = self._activate(z2)
        logits = self._matmul(h2, self.w3) + self.b3
        self.cache = {
            "inputs": inputs,
            "z1": z1,
            "h1": h1,
            "z2": z2,
            "h2": h2,
            "logits": logits,
        }
        return logits

    def compute_loss(self, inputs: Any, targets: Any, weight_decay: float = 0.0) -> float:
        """Compute the current loss without updating gradients."""
        logits = self.forward(inputs)
        loss, _ = self._softmax_loss(logits, targets)
        reg = 0.5 * weight_decay * (
            self.xp.sum(self.w1 * self.w1)
            + self.xp.sum(self.w2 * self.w2)
            + self.xp.sum(self.w3 * self.w3)
        )
        return float(to_numpy(loss + reg))

    def loss_and_backward(self, inputs: Any, targets: Any, weight_decay: float) -> float:
        """Run forward, compute loss, and back-propagate gradients."""
        logits = self.forward(inputs)
        loss, grad_logits = self._softmax_loss(logits, targets)
        self._backward(grad_logits, weight_decay)
        reg = 0.5 * weight_decay * (
            self.xp.sum(self.w1 * self.w1)
            + self.xp.sum(self.w2 * self.w2)
            + self.xp.sum(self.w3 * self.w3)
        )
        return float(to_numpy(loss + reg))

    def predict(self, inputs: Any) -> Any:
        """Predict labels for a batch of features."""
        logits = self.forward(inputs)
        return self.xp.argmax(logits, axis=1)

    def step(self, learning_rate: float, grad_clip: float = 5.0) -> None:
        """Apply one SGD update."""
        clipped = {name: self.xp.clip(grad, -grad_clip, grad_clip) for name, grad in self.grads.items()}
        # 这里做一个很轻的梯度裁剪，避免小批量训练时数值突然发散。
        self.w1 -= learning_rate * clipped["w1"]
        self.b1 -= learning_rate * clipped["b1"]
        self.w2 -= learning_rate * clipped["w2"]
        self.b2 -= learning_rate * clipped["b2"]
        self.w3 -= learning_rate * clipped["w3"]
        self.b3 -= learning_rate * clipped["b3"]

    def save(self, checkpoint_path: Path) -> None:
        """Save model weights and metadata to an .npz file."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_dim2": self.hidden_dim2,
            "output_dim": self.output_dim,
            "activation": self.activation,
        }
        np.savez(
            checkpoint_path,
            metadata=json.dumps(metadata, ensure_ascii=False),
            w1=to_numpy(self.w1),
            b1=to_numpy(self.b1),
            w2=to_numpy(self.w2),
            b2=to_numpy(self.b2),
            w3=to_numpy(self.w3),
            b3=to_numpy(self.b3),
        )

    @classmethod
    def load(cls, checkpoint_path: Path, xp: Any) -> "ThreeLayerMLP":
        """Load a model checkpoint."""
        payload = np.load(checkpoint_path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata"]))
        model = cls(
            input_dim=metadata["input_dim"],
            hidden_dim=metadata["hidden_dim"],
            hidden_dim2=metadata.get("hidden_dim2", metadata["hidden_dim"]),
            output_dim=metadata["output_dim"],
            activation=metadata["activation"],
            xp=xp,
        )
        model.w1 = xp.asarray(payload["w1"])
        model.b1 = xp.asarray(payload["b1"])
        model.w2 = xp.asarray(payload["w2"])
        model.b2 = xp.asarray(payload["b2"])
        model.w3 = xp.asarray(payload["w3"])
        model.b3 = xp.asarray(payload["b3"])
        return model

    def _backward(self, grad_logits: Any, weight_decay: float) -> None:
        """Back-propagate gradients through the network."""
        h2 = self.cache["h2"]
        h1 = self.cache["h1"]
        z2 = self.cache["z2"]
        z1 = self.cache["z1"]
        inputs = self.cache["inputs"]

        self.grads["w3"] = self._matmul(h2.T, grad_logits) + weight_decay * self.w3
        self.grads["b3"] = self.xp.sum(grad_logits, axis=0)

        grad_h2 = self._matmul(grad_logits, self.w3.T)
        grad_z2 = grad_h2 * self._activation_grad(z2)
        self.grads["w2"] = self._matmul(h1.T, grad_z2) + weight_decay * self.w2
        self.grads["b2"] = self.xp.sum(grad_z2, axis=0)

        grad_h1 = self._matmul(grad_z2, self.w2.T)
        grad_z1 = grad_h1 * self._activation_grad(z1)
        self.grads["w1"] = self._matmul(inputs.T, grad_z1) + weight_decay * self.w1
        self.grads["b1"] = self.xp.sum(grad_z1, axis=0)

    def _softmax_loss(self, logits: Any, targets: Any) -> tuple[Any, Any]:
        """Compute softmax cross-entropy and the gradient on logits."""
        shifted = logits - self.xp.max(logits, axis=1, keepdims=True)
        exp_scores = self.xp.exp(shifted)
        probs = exp_scores / self.xp.sum(exp_scores, axis=1, keepdims=True)
        batch_indices = self.xp.arange(targets.shape[0])
        losses = -self.xp.log(probs[batch_indices, targets] + 1e-12)
        loss = self.xp.mean(losses)

        grad_logits = probs.copy()
        grad_logits[batch_indices, targets] -= 1.0
        grad_logits /= targets.shape[0]
        return loss, grad_logits

    def _activate(self, inputs: Any) -> Any:
        """Apply the configured activation."""
        if self.activation == "relu":
            return self.xp.maximum(inputs, 0.0)
        if self.activation == "tanh":
            return self.xp.tanh(inputs)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + self.xp.exp(-inputs))
        raise ValueError(f"不支持的激活函数: {self.activation}")

    def _activation_grad(self, inputs: Any) -> Any:
        """Derivative of the configured activation."""
        if self.activation == "relu":
            return (inputs > 0).astype(inputs.dtype)
        if self.activation == "tanh":
            activated = self.xp.tanh(inputs)
            return 1.0 - activated * activated
        if self.activation == "sigmoid":
            activated = 1.0 / (1.0 + self.xp.exp(-inputs))
            return activated * (1.0 - activated)
        raise ValueError(f"不支持的激活函数: {self.activation}")

    def _matmul(self, left: Any, right: Any) -> Any:
        """Compute matrix multiplication and suppress known NumPy matmul warnings."""
        if self.xp is np:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
                return left @ right
        return left @ right
