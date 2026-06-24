"""Fast GPU out-of-fold supervised grounding head (TileTalk's learnable part).

Given frozen features and a per-query binary relevance matrix, fit a per-query
linear (logistic) or shared-trunk MLP head that maps H&E-derived features to
relevance, scoring every cell with K-fold out-of-fold prediction (no leakage).
All queries are trained jointly as independent output logits with per-query
class balancing, so the full grid runs in seconds on a GPU rather than the
~25 min a per-query scikit-learn loop took.
"""
from __future__ import annotations

import numpy as np


def _device(prefer: str = "cuda") -> str:
    import torch
    return prefer if torch.cuda.is_available() else "cpu"


def oof_scores(feat: np.ndarray, Y: np.ndarray, head: str = "logistic",
               nfolds: int = 5, seed: int = 0, epochs: int = 400,
               lr: float = 5e-2, weight_decay: float = 1e-3, hidden: int = 256,
               device: str = "cuda") -> np.ndarray:
    """Out-of-fold relevance scores.

    feat: [N, D] frozen features. Y: [N, Q] binary relevance per query.
    Returns scores [Q, N] (held-out logits; higher = more relevant).
    """
    import torch
    from sklearn.model_selection import KFold

    dev = _device(device)
    N, D = feat.shape
    Q = Y.shape[1]
    Xt = torch.tensor(feat, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    out = np.zeros((Q, N), dtype=np.float32)
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    g = torch.Generator(device="cpu").manual_seed(seed)

    for tr, te in kf.split(feat):
        tr_i = torch.tensor(tr, device=dev)
        te_i = torch.tensor(te, device=dev)
        Xtr, Ytr = Xt[tr_i], Yt[tr_i]
        mu = Xtr.mean(0, keepdim=True)
        sd = Xtr.std(0, keepdim=True) + 1e-6
        Xtr_n = (Xtr - mu) / sd
        Xte_n = (Xt[te_i] - mu) / sd

        model = _make_model(head, D, Q, hidden, seed, dev)
        _train(model, Xtr_n, Ytr, head, epochs, lr, weight_decay)
        model.eval()
        with torch.no_grad():
            pred = model(Xte_n)              # [te, Q]
        out[:, te] = pred.t().cpu().numpy()
    return out


def _make_model(head, D, Q, hidden, seed, dev):
    import torch
    torch.manual_seed(seed)
    if head == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(D, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(hidden, Q)).to(dev)
    return torch.nn.Linear(D, Q).to(dev)


def _train(model, X, Y, head, epochs, lr, weight_decay):
    """Convex logistic head -> LBFGS (matches scikit-learn); MLP -> Adam."""
    import torch
    pos = Y.sum(0)
    pos_weight = ((Y.shape[0] - pos) / (pos + 1e-6)).clamp(1.0, 1000.0)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    params = list(model.parameters())
    model.train()
    if head == "mlp":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad(); loss = lossf(model(X), Y); loss.backward(); opt.step()
    else:
        # max_iter=200 matches the released cellseek score artifacts; the convex
        # logistic is well fit by ~60 iters too (≈equal metrics, ~5s/fold) if speed
        # is preferred.
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=200, history_size=10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = lossf(model(X), Y)
            loss = loss + weight_decay * sum(p.pow(2).sum() for p in params)
            loss.backward()
            return loss
        opt.step(closure)


def fit_predict(feat_train: np.ndarray, Y_train: np.ndarray, feat_test: np.ndarray,
                head: str = "logistic", seed: int = 0, epochs: int = 400,
                lr: float = 5e-2, weight_decay: float = 1e-3, hidden: int = 256,
                device: str = "cuda") -> np.ndarray:
    """Train on one sample, predict another (cross-sample transfer).

    Returns scores [Q, N_test]. Standardization uses train statistics.
    """
    import torch
    dev = _device(device)
    D = feat_train.shape[1]
    Q = Y_train.shape[1]
    Xtr = torch.tensor(feat_train, dtype=torch.float32, device=dev)
    Ytr = torch.tensor(Y_train, dtype=torch.float32, device=dev)
    Xte = torch.tensor(feat_test, dtype=torch.float32, device=dev)
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    model = _make_model(head, D, Q, hidden, seed, dev)
    _train(model, Xtr_n, Ytr, head, epochs, lr, weight_decay)
    model.eval()
    with torch.no_grad():
        return model(Xte_n).t().cpu().numpy()


def relevance_matrix(qs, pool, expr, gene_names) -> np.ndarray:
    """[N, Q] binary relevance for all queries (uses queries.relevance_mask)."""
    from . import queries as Q
    cols = [Q.relevance_mask(q, pool, expr, gene_names).astype(np.float32)
            for _, q in qs.iterrows()]
    return np.stack(cols, axis=1)
