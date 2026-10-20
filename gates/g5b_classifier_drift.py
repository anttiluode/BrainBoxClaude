"""G5b - First AI-level test: do conserved block totals protect a TRAINED network from drift?

The passive-medium result (G3b, G4): keep each interface's total, let the split inside
drift, and the function survives — to the degree the inside is "well mixed" relative to
the outside (G4 ratio law). The classifier analogue of "well mixed" is that inputs
inside a block are correlated: then sum_j W_ij x_j ~ (sum_{j in block} W_ij) * xbar_block,
so the block total carries most of what the unit computes from that block.

Setup: sklearn digits (8x8, 1797 images), MLP 64 -> 64 (tanh) -> 10, trained in numpy
(full-batch Adam). After training, the first-layer weights drift:
    w <- w * exp(s * xi)      (multiplicative, sign-preserving, independent per weight)
Repair = for each hidden unit and each group of input pixels, add the same correction to
every weight in the group so the group's sum returns to its stored post-training value
(minimum-norm correction that restores the total). Stored numbers per repair:
    row      : 1 group of 64 per unit        ->   64 stored totals
    block4x4 : 4 local 4x4 pixel blocks      ->  256
    block2x2 : 16 local 2x2 pixel blocks     -> 1024
    rand4    : 16 RANDOM groups of 4 pixels  -> 1024   (matched storage, no locality)
    oracle   : restore every weight          -> 4096
    snap8    : restore from an 8-bit copy of all weights  (bits = 1024 float32 totals)
    snap2    : restore from a 2-bit copy of all weights   (bits =  256 float32 totals)
               (bit-matched controls added after the first run: a digital system can just
                keep a quantized snapshot; block totals must beat THAT to matter for AI)
Two input orders: natural, and pixels PERMUTED by a fixed permutation before training
(so index-local "blocks" contain unrelated pixels).

PREDICTIONS (before running):
 Q1. On natural inputs, block2x2 removes far more drift damage than rand4 at identical
     storage (locality is what matters, not the number of stored totals).
 Q2. On permuted inputs, block2x2 loses its advantage (falls to about rand4's level).
 Q3. row (64 numbers) gives little; block4x4 sits between row and block2x2.
If Q1 fails, interface-total conservation is not an AI-relevant principle for drift.
"""
import json, os
import numpy as np
from sklearn.datasets import load_digits

H = 64
EPOCHS = 400
SEEDS = [0, 1, 2]
DRIFT = [0.3, 0.6, 1.0]
DRAWS = 20


def data(perm=None):
    d = load_digits()
    X = d.data / 16.0
    if perm is not None:
        X = X[:, perm]
    y = d.target
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(X))
    tr, te = idx[:1260], idx[1260:]
    return X[tr], y[tr], X[te], y[te]


def train(X, y, seed):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((H, 64)) * np.sqrt(1 / 64); b1 = np.zeros(H)
    W2 = rng.standard_normal((10, H)) * np.sqrt(1 / H); b2 = np.zeros(10)
    P = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
    Y = np.eye(10)[y]
    for t in range(1, EPOCHS + 1):
        h = np.tanh(X @ W1.T + b1)
        z = h @ W2.T + b2
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        dz = (p - Y) / len(X)
        gW2 = dz.T @ h; gb2 = dz.sum(0)
        dh = (dz @ W2) * (1 - h ** 2)
        gW1 = dh.T @ X; gb1 = dh.sum(0)
        G = [gW1, gb1, gW2, gb2]
        for k in range(4):
            m[k] = 0.9 * m[k] + 0.1 * G[k]
            v[k] = 0.999 * v[k] + 0.001 * G[k] ** 2
            P[k] -= 0.01 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    return P


def acc(P, X, y):
    W1, b1, W2, b2 = P
    return float(((np.tanh(X @ W1.T + b1) @ W2.T + b2).argmax(1) == y).mean())


def groups(kind, rng):
    """List of index arrays over the 64 input positions (row-major 8x8 index grid)."""
    grid = np.arange(64).reshape(8, 8)
    if kind == "row":
        return [np.arange(64)]
    if kind == "block4x4":
        return [grid[i:i + 4, j:j + 4].ravel() for i in (0, 4) for j in (0, 4)]
    if kind == "block2x2":
        return [grid[i:i + 2, j:j + 2].ravel() for i in range(0, 8, 2) for j in range(0, 8, 2)]
    if kind == "rand4":
        p = rng.permutation(64)
        return [p[k:k + 4] for k in range(0, 64, 4)]
    raise ValueError(kind)


def quantize(W0, bits):
    """Per-row uniform quantization of W0 to `bits` bits (row min/max kept, negligible cost)."""
    lo, hi = W0.min(1, keepdims=True), W0.max(1, keepdims=True)
    q = np.round((W0 - lo) / (hi - lo) * (2 ** bits - 1))
    return lo + q * (hi - lo) / (2 ** bits - 1)


def repair(W, W0, grp):
    W = W.copy()
    for g in grp:
        d = (W0[:, g].sum(1) - W[:, g].sum(1)) / len(g)
        W[:, g] += d[:, None]
    return W


def run():
    rng = np.random.default_rng(999)
    perm = rng.permutation(64)
    kinds = ["none", "row", "block4x4", "block2x2", "rand4", "snap2", "snap8", "oracle"]
    res = {"config": dict(H=H, epochs=EPOCHS, seeds=SEEDS, drift=DRIFT, draws=DRAWS,
                          storage={"row": 64, "block4x4": 256, "block2x2": 1024, "rand4": 1024, "oracle": 4096,
                                                    "snap2_bits_as_floats": 256, "snap8_bits_as_floats": 1024})}
    for order in ["natural", "permuted"]:
        Xtr, ytr, Xte, yte = data(None if order == "natural" else perm)
        # within-group input correlation (the classifier's "well-mixed" measure)
        C = np.corrcoef(Xtr.T + 1e-9 * rng.standard_normal(Xtr.T.shape))
        def mean_within(grp):
            vals = [C[np.ix_(g, g)][np.triu_indices(len(g), 1)].mean() for g in grp]
            return float(np.nanmean(vals))
        res[order] = {"within_group_corr": {k: mean_within(groups(k, rng)) for k in ["block2x2", "rand4"]}}
        for seed in SEEDS:
            P = train(Xtr, ytr, seed)
            clean = acc(P, Xte, yte)
            W0 = P[0].copy()
            for s in DRIFT:
                out = {k: [] for k in kinds}
                for _ in range(DRAWS):
                    Wd = W0 * np.exp(s * rng.standard_normal(W0.shape))
                    for k in kinds:
                        if k == "none":
                            W = Wd
                        elif k == "oracle":
                            W = W0
                        elif k.startswith("snap"):
                            W = quantize(W0, int(k[4:]))
                        else:
                            W = repair(Wd, W0, groups(k, rng))
                        out[k].append(acc([W] + P[1:], Xte, yte))
                res[order].setdefault(f"s{s}", []).append({"clean": clean, **{k: float(np.mean(v)) for k, v in out.items()}})
        print(f"\n{order.upper()} inputs  (mean within-group pixel correlation: 2x2 blocks "
              f"{res[order]['within_group_corr']['block2x2']:.2f}, random groups {res[order]['within_group_corr']['rand4']:.2f})")
        print(f"  {'drift':6s} " + " ".join(f"{k:>9s}" for k in kinds) + "   damage removed: " + " ".join(f"{k:>8s}" for k in kinds[1:7]))
        for s in DRIFT:
            rows = res[order][f"s{s}"]
            mean = {k: np.mean([r[k] for r in rows]) for k in kinds}
            dmg = mean["oracle"] - mean["none"]
            rem = {k: (mean[k] - mean["none"]) / dmg if dmg > 1e-9 else float("nan") for k in kinds[1:7]}
            res[order][f"summary_s{s}"] = {"acc": mean, "removed": rem}
            print(f"  s={s:<4} " + " ".join(f"{mean[k]:9.3f}" for k in kinds) + "                  " +
                  " ".join(f"{rem[k]:+8.2f}" for k in kinds[1:7]))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g5b_classifier_drift.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
