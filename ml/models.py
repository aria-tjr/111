"""
NEXUS ML Model Definitions.

Two models:
  1. RandomForestSignalClassifier  — sklearn ensemble, predicts LONG/SHORT/NEUTRAL
  2. LSTMPricePredictor            — PyTorch sequence model, predicts next-bar direction

Both models expose a common interface:
  .predict(X) -> np.ndarray of class probabilities [p_short, p_neutral, p_long]
  .predict_proba(X) -> same as predict
  .is_trained -> bool

PyTorch is optional. If not installed the LSTM class degrades gracefully.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────
try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — RandomForest model unavailable")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — LSTM model unavailable")


# ──────────────────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────────────────

LABEL_SHORT   = 0
LABEL_NEUTRAL = 1
LABEL_LONG    = 2
LABEL_NAMES   = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}


# ──────────────────────────────────────────────────────────
# 1. Random Forest Ensemble Classifier
# ──────────────────────────────────────────────────────────


class RandomForestSignalClassifier:
    """
    Ensemble of RandomForest + GradientBoosting via soft VotingClassifier.

    Input  : feature vector(s) of shape (FEATURE_DIM,) or (N, FEATURE_DIM)
    Output : probabilities [p_short, p_neutral, p_long], shape (3,) or (N, 3)
    """

    def __init__(self):
        self.scaler: Optional[object] = None
        self.clf: Optional[object] = None
        self.is_trained: bool = False
        self._feature_importances: Optional[np.ndarray] = None

    def build(self) -> "RandomForestSignalClassifier":
        """Instantiate (untrained) model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for RandomForestSignalClassifier")

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.clf = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
            weights=[2, 1],  # RF gets more weight (more robust)
        )
        self.scaler = StandardScaler()
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestSignalClassifier":
        """Train the model. X: (N, FEATURE_DIM), y: (N,) with values 0/1/2."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        if self.clf is None:
            self.build()

        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_trained = True

        # Store feature importances from RF sub-estimator
        try:
            rf_est = self.clf.estimators_[0]
            self._feature_importances = rf_est.feature_importances_
        except Exception:
            pass

        logger.info("RandomForestSignalClassifier trained on %d samples", len(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability array shape (3,) for single sample or (N, 3) for batch.
        Order: [p_short, p_neutral, p_long]
        """
        if not self.is_trained or self.clf is None:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)
        proba = self.clf.predict_proba(X_scaled).astype(np.float32)

        # VotingClassifier may reorder classes; align to [SHORT, NEUTRAL, LONG]
        classes = list(self.clf.classes_)
        ordered = np.zeros((len(X), 3), dtype=np.float32)
        for i, cls in enumerate(classes):
            if 0 <= cls <= 2:
                ordered[:, cls] = proba[:, i]
        # Normalize rows
        row_sums = ordered.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        ordered /= row_sums

        return ordered[0] if single else ordered

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "scaler": self.scaler,
                         "fi": self._feature_importances}, f)
        logger.info("RandomForest model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "RandomForestSignalClassifier":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.clf    = data["clf"]
        obj.scaler = data["scaler"]
        obj._feature_importances = data.get("fi")
        obj.is_trained = True
        logger.info("RandomForest model loaded from %s", path)
        return obj

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        return self._feature_importances


# ──────────────────────────────────────────────────────────
# 2. LSTM Price Direction Predictor
# ──────────────────────────────────────────────────────────


if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        """
        Bidirectional LSTM → attention → FC for 3-class classification.

        Architecture:
          Input  : (batch, seq_len, input_dim)
          LSTM   : 2 layers, bidirectional, hidden=128
          Attention: self-attention over time steps
          FC     : 256 → 128 → 3
        """

        def __init__(self, input_dim: int, hidden_dim: int = 128,
                     num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            lstm_out_dim = hidden_dim * 2  # bidirectional

            # Attention
            self.attn_w = nn.Linear(lstm_out_dim, 1)

            self.fc = nn.Sequential(
                nn.Linear(lstm_out_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(128, 3),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq, features)
            out, _ = self.lstm(x)               # (batch, seq, hidden*2)

            # Self-attention: compute weights over time steps
            attn_scores = self.attn_w(out)      # (batch, seq, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (out * attn_weights).sum(dim=1)  # (batch, hidden*2)

            logits = self.fc(context)           # (batch, 3)
            return logits


class LSTMPricePredictor:
    """
    PyTorch LSTM-based price direction predictor.

    Input  : sequence of feature vectors, shape (seq_len, FEATURE_DIM)
    Output : probabilities [p_short, p_neutral, p_long], shape (3,)
    """

    def __init__(self, seq_len: int = 50, input_dim: int = 40,
                 hidden_dim: int = 128, num_layers: int = 2):
        self.seq_len    = seq_len
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.net        = None
        self.scaler     = None
        self.is_trained = False
        self.device     = "cpu"

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed — LSTMPricePredictor unavailable")

    def build(self) -> "LSTMPricePredictor":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTMPricePredictor")
        self.net = _LSTMNet(self.input_dim, self.hidden_dim, self.num_layers)
        self.net.to(self.device)
        return self

    def fit(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_split: float = 0.1,
    ) -> "LSTMPricePredictor":
        """
        Train the LSTM.

        X_seq : (N, seq_len, input_dim)
        y     : (N,) with values 0/1/2
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        if self.net is None:
            self.build()

        # Feature-wise normalization across the training set
        flat = X_seq.reshape(-1, self.input_dim).astype(np.float32)
        self.mean_ = flat.mean(axis=0)
        self.std_  = flat.std(axis=0) + 1e-8

        X_norm = ((X_seq - self.mean_) / self.std_).astype(np.float32)
        y_t    = torch.from_numpy(y.astype(np.int64))
        X_t    = torch.from_numpy(X_norm)

        # Train / val split
        n = len(y)
        n_val = max(1, int(n * val_split))
        n_tr  = n - n_val
        X_tr, X_val = X_t[:n_tr], X_t[n_tr:]
        y_tr, y_val = y_t[:n_tr], y_t[n_tr:]

        # Class weights for imbalanced labels
        counts = np.bincount(y, minlength=3).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        weights = torch.tensor(1.0 / counts).to(self.device)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_val_loss = float("inf")
        best_state    = None

        self.net.train()
        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n_tr)
            X_tr, y_tr = X_tr[perm], y_tr[perm]

            epoch_loss = 0.0
            for start in range(0, n_tr, batch_size):
                xb = X_tr[start: start + batch_size].to(self.device)
                yb = y_tr[start: start + batch_size].to(self.device)
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            scheduler.step()

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(X_val.to(self.device))
                val_loss   = criterion(val_logits, y_val.to(self.device)).item()
            self.net.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                logger.info("LSTM epoch %d/%d — train_loss=%.4f val_loss=%.4f",
                            epoch + 1, epochs, epoch_loss / n_tr, val_loss)

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        self.is_trained = True
        logger.info("LSTMPricePredictor trained on %d samples (best val_loss=%.4f)",
                    n_tr, best_val_loss)
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """
        X_seq : (seq_len, input_dim) or (N, seq_len, input_dim)
        Returns: (3,) or (N, 3) probabilities [SHORT, NEUTRAL, LONG]
        """
        if not TORCH_AVAILABLE or not self.is_trained or self.net is None:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

        single = X_seq.ndim == 2
        if single:
            X_seq = X_seq[np.newaxis]  # (1, seq, feat)

        X_norm = ((X_seq - self.mean_) / self.std_).astype(np.float32)
        X_t    = torch.from_numpy(X_norm).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_t)
            proba  = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

        return proba[0] if single else proba

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_seq)

    def save(self, path: str):
        if not TORCH_AVAILABLE or not self.is_trained:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "mean":       self.mean_,
            "std":        self.std_,
            "seq_len":    self.seq_len,
            "input_dim":  self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }, path)
        logger.info("LSTM model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LSTMPricePredictor":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(
            seq_len=ckpt["seq_len"],
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
        )
        obj.build()
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        obj.mean_       = ckpt["mean"]
        obj.std_        = ckpt["std"]
        obj.is_trained  = True
        logger.info("LSTM model loaded from %s", path)
        return obj
