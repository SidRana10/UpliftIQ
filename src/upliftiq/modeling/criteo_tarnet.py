"""TARNet for the binary-treatment Criteo experiment (PyTorch).

Architecture (Shalit et al., 2017): a shared trunk of fully connected ReLU layers on the scaled
f0-f11, then two outcome heads, one for control (t = 0) and one for treatment (t = 1), each a
small MLP producing a visit logit. Training minimizes binary cross-entropy of the *assigned*
treatment's head on the observed outcome only (the counterfactual head gets no gradient from that
row). The predicted uplift is sigmoid(head1) - sigmoid(head0).

Same interface as ``modeling.criteo_learners``: ``fit``, ``predict_uplift``, ``predict_outcomes``
(n, 2), ``recommend(X, budget=None, row_ids=None)``. Preprocessing is the train-only clip and
standardize scaler of C0, fitted on the rows given to ``fit`` (excluding its early-stopping rows).
Training is mini-batch with the features held once as a float32 tensor; only one batch at a time is
moved to the accelerator. Fits are bit-reproducible on CPU for a fixed seed; Apple MPS is seeded
but GPU kernels may differ in the last digits.
"""

from __future__ import annotations

import contextlib
import copy
import os
import time

# LightGBM, scikit-learn and PyTorch each bundle an OpenMP runtime. On macOS, PyTorch's runtime
# running a multi-threaded pool in the same process as LightGBM's crashes or deadlocks. Keeping
# PyTorch's pool at one thread (set before torch loads) avoids it; the MLP is small.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch import nn  # noqa: E402

from upliftiq.data.criteo_prepare import fit_scaler  # noqa: E402
from upliftiq.modeling.criteo_learners import N_FEATURES, SEED, UpliftModel  # noqa: E402

PRED_BATCH = 65_536


@contextlib.contextmanager
def _single_thread():
    """Run torch CPU ops on one thread (see the OpenMP note at the top of the module)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def resolve_device(device: str = "auto") -> torch.device:
    """MPS when available and working, otherwise CPU; ``cpu``/``mps`` force a choice."""
    if device == "cpu":
        return torch.device("cpu")
    if device not in ("auto", "mps"):
        raise ValueError("device must be 'auto', 'cpu' or 'mps'")
    try:
        if torch.backends.mps.is_available():
            torch.zeros(1, device="mps") + 1  # fails fast if the backend is unusable
            return torch.device("mps")
    except (RuntimeError, AttributeError):
        pass
    if device == "mps":
        raise RuntimeError("MPS requested but not available")
    return torch.device("cpu")


class TARNetModule(nn.Module):
    def __init__(
        self,
        n_in: int = N_FEATURES,
        hidden: int = 128,
        depth: int = 2,
        head_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
            for _ in range(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(n, 2) logits: column a is the head for treatment a."""
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


def factual_loss(logits: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean BCE of the head matching each row's assigned treatment."""
    picked = logits.gather(1, t.long()[:, None])[:, 0]
    return nn.functional.binary_cross_entropy_with_logits(picked, y)


class TARNet(UpliftModel):
    name = "tarnet"

    def __init__(
        self,
        seed: int = SEED,
        params: dict | None = None,
        n_jobs: int = 4,
        hidden: int = 128,
        depth: int = 2,
        head_hidden: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 4096,
        max_epochs: int = 30,
        patience: int = 3,
        val_fraction: float = 0.1,
        device: str = "auto",
    ):
        super().__init__(seed, params, n_jobs)
        self.arch = {
            "hidden": hidden,
            "depth": depth,
            "head_hidden": head_hidden,
            "dropout": dropout,
        }
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.val_fraction = val_fraction
        self.device_request = device

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X, np.arange(len(X)))

    def _fit(self, X, t, y):
        with _single_thread():
            self._fit_impl(X, t, y)

    def _fit_impl(self, X, t, y):
        start = time.perf_counter()
        torch.manual_seed(self.seed)
        self.device_ = resolve_device(self.device_request)
        tr, va = train_test_split(
            np.arange(len(y)),
            test_size=self.val_fraction,
            random_state=self.seed,
            stratify=t * 2 + y.astype(int),
        )
        self.scaler_ = fit_scaler(X, tr)  # statistics from the fitting rows only
        Z = torch.from_numpy(self._scale(X))
        T = torch.from_numpy(t.astype(np.float32))
        Y = torch.from_numpy(y.astype(np.float32))
        net = TARNetModule(**self.arch).to(self.device_)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)
        tr_t, va_t = torch.from_numpy(tr), torch.from_numpy(va)
        self.history_, best, best_state, bad = [], np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            e0 = time.perf_counter()
            net.train()
            perm = tr_t[torch.randperm(len(tr_t), generator=gen)]
            total = 0.0
            for s in range(0, len(perm), self.batch_size):
                b = perm[s : s + self.batch_size]
                loss = factual_loss(
                    net(Z[b].to(self.device_)), T[b].to(self.device_), Y[b].to(self.device_)
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            val = self._loss(net, Z, T, Y, va_t)
            self.history_.append(
                {
                    "epoch": epoch,
                    "train_loss": total / len(perm),
                    "val_loss": val,
                    "seconds": time.perf_counter() - e0,
                }
            )
            if val < best - 1e-6:
                best, bad, self.best_epoch_ = val, 0, epoch
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net_ = net.eval()
        self.fit_seconds_ = time.perf_counter() - start

    @torch.no_grad()
    def _loss(self, net, Z, T, Y, rows) -> float:
        net.eval()
        total = 0.0
        for s in range(0, len(rows), PRED_BATCH):
            b = rows[s : s + PRED_BATCH]
            batch_loss = factual_loss(
                net(Z[b].to(self.device_)), T[b].to(self.device_), Y[b].to(self.device_)
            )
            total += batch_loss.item() * len(b)
        return total / len(rows)

    @torch.no_grad()
    def predict_outcomes(self, X) -> np.ndarray:
        """(n, 2): P(visit | x, control) and P(visit | x, treatment)."""
        start = time.perf_counter()
        Z = self._scale(self._check(X))
        out = np.empty((len(Z), 2), dtype=np.float64)
        with _single_thread():
            for s in range(0, len(Z), PRED_BATCH):
                z = torch.from_numpy(Z[s : s + PRED_BATCH]).to(self.device_)
                out[s : s + PRED_BATCH] = torch.sigmoid(self.net_(z)).cpu().numpy()
        self.last_predict_seconds_ = time.perf_counter() - start
        self.last_predict_rows_ = len(Z)
        return out

    def training_summary(self) -> dict:
        """Numbers kept for later benchmarking."""
        ms_per_1k = None
        if hasattr(self, "last_predict_seconds_"):
            ms_per_1k = 1e6 * self.last_predict_seconds_ / max(self.last_predict_rows_, 1)
        return {
            "device": str(self.device_),
            "epochs_run": len(self.history_),
            "best_epoch": self.best_epoch_,
            "fit_seconds": self.fit_seconds_,
            "best_val_loss": min(h["val_loss"] for h in self.history_),
            "last_predict_ms_per_1k_rows": ms_per_1k,
        }
