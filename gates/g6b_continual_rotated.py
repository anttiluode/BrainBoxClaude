"""G6b - Continual learning where forgetting is real: rotated digits, shared head.

Why G6b exists: G6 (split digits, one head per task) sat at the ceiling — plain finetuning
ended at 0.990 with forgetting -0.007, so no method could be distinguished. Redesigned
BEFORE rerunning, with the same verdict rule:

Benchmark: domain-incremental rotated digits. Five tasks = all 10 digit classes, images
rotated by 0, 30, 60, 90, 120 degrees, learned in sequence. ONE shared 10-way head.
Network 64 -> 32 (tanh, no bias) -> 10. SGD + momentum.

Isolating the claim: every projection method protects the HEAD identically (GPM on the
hidden activations, 95% energy per task). They differ ONLY in the input-layer protected
subspace:
  gpm      : learned from data, rank grows per task (stores the basis)
  gpm16    : learned, rank capped at 16 (stores 64x16)
  pyramid2x2: geometric — zero-sum inside local 2x2 pixel blocks after task 1 (rank 16, stores nothing)
  pyramid4x4: geometric, 4x4 blocks (rank 4, stores nothing)
  rand16   : zero-sum inside 16 random groups of 4 pixels (rank 16, stores nothing)
plus finetune and EWC (both layers, lambda by final ACC from a grid — optimistic for EWC).

VERDICT RULE (unchanged): pyramid2x2 counts as AI-relevant only if its final ACC on these
natural (un-permuted) images is >= gpm16's.
"""
import json, os
import numpy as np
from scipy.ndimage import rotate
from sklearn.datasets import load_digits

H = 32
EPOCHS = 300
LR = 0.1
MOM = 0.9
SEEDS = range(5)
ANGLES = [0, 30, 60, 90, 120]
EWC_GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0]
METHODS = ["finetune", "ewc", "gpm", "gpm16", "pyramid2x2", "pyramid4x4", "rand16"]


def load():
    d = load_digits()
    imgs = d.images / 16.0
    rng = np.random.default_rng(321)
    idx = rng.permutation(len(imgs)); cut = int(0.7 * len(imgs))
    tr, te = idx[:cut], idx[cut:]
    tasks = []
    for a in ANGLES:
        R = np.array([rotate(im, a, reshape=False, order=1, mode="constant") for im in imgs]).reshape(len(imgs), 64)
        R = np.clip(R, 0, 1)
        tasks.append((R[tr], d.target[tr], R[te], d.target[te]))
    return tasks


def group_basis(kind, rng):
    grid = np.arange(64).reshape(8, 8)
    if kind == "2x2":
        G = [grid[i:i + 2, j:j + 2].ravel() for i in range(0, 8, 2) for j in range(0, 8, 2)]
    elif kind == "4x4":
        G = [grid[i:i + 4, j:j + 4].ravel() for i in (0, 4) for j in (0, 4)]
    else:
        p = rng.permutation(64); G = [p[k:k + 4] for k in range(0, 64, 4)]
    B = np.zeros((64, len(G)))
    for c, g in enumerate(G):
        B[g, c] = 1 / np.sqrt(len(g))
    return B


def forward(W1, V, X):
    h = np.tanh(X @ W1.T)
    z = h @ V.T; z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return h, p


def grads(W1, V, X, y):
    h, p = forward(W1, V, X)
    dz = (p - np.eye(10)[y]) / len(X)
    gV = dz.T @ h
    gW1 = ((dz @ V) * (1 - h ** 2)).T @ X
    return gW1, gV


def acc(W1, V, X, y):
    return float((forward(W1, V, X)[1].argmax(1) == y).mean())


def gpm_extend(U, A, cap=None, thr=0.95):
    """Add basis vectors for the energy of A (rows = samples) not yet in span(U)."""
    total = (A ** 2).sum()
    Ar = A - (A @ U) @ U.T
    u, s, _ = np.linalg.svd(Ar.T, full_matrices=False)
    k = 0; e = ((A @ U) ** 2).sum()
    while k < len(s) and e < thr * total:
        e += s[k] ** 2; k += 1
    if cap is not None:
        k = max(0, min(k, cap - U.shape[1]))
    return np.concatenate([U, u[:, :k]], 1) if k > 0 else U


def run_method(method, tasks, seed, lam=0.0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((H, 64)) * np.sqrt(1 / 64)
    V = rng.standard_normal((10, H)) * np.sqrt(1 / H)
    Ux = np.zeros((64, 0)); Uh = np.zeros((H, 0))
    F1 = np.zeros_like(W1); FV = np.zeros_like(V); A1 = W1.copy(); AV = V.copy()
    T = len(tasks); R = np.zeros((T, T))
    proj = method not in ("finetune", "ewc")
    for t, (Xtr, ytr, _, _) in enumerate(tasks):
        m1 = np.zeros_like(W1); mV = np.zeros_like(V)
        for _ in range(EPOCHS):
            g1, gV = grads(W1, V, Xtr, ytr)
            if method == "ewc" and t > 0:
                g1 = g1 + lam * F1 * (W1 - A1); gV = gV + lam * FV * (V - AV)
            if proj:
                if Ux.shape[1]: g1 = g1 - (g1 @ Ux) @ Ux.T
                if Uh.shape[1]: gV = gV - (gV @ Uh) @ Uh.T
            m1 = MOM * m1 + g1; mV = MOM * mV + gV
            W1 -= LR * m1; V -= LR * mV
        for j in range(t + 1):
            R[t, j] = acc(W1, V, tasks[j][2], tasks[j][3])
        h, p = forward(W1, V, Xtr)
        if method == "ewc":
            dz = p - np.eye(10)[ytr]
            dh = (dz @ V) * (1 - h ** 2)
            F1 += np.einsum("nh,nd->hd", dh ** 2, Xtr ** 2) / len(Xtr)
            FV += np.einsum("nc,nh->ch", dz ** 2, h ** 2) / len(Xtr)
            A1 = W1.copy(); AV = V.copy()
        if proj:
            Uh = gpm_extend(Uh, h)                                  # head: identical for all
            if method == "gpm":
                Ux = gpm_extend(Ux, Xtr)
            elif method == "gpm16":
                Ux = gpm_extend(Ux, Xtr, cap=16)
            elif t == 0:
                Ux = group_basis({"pyramid2x2": "2x2", "pyramid4x4": "4x4", "rand16": "rand"}[method], rng)
    stored = {"gpm": Ux.size, "gpm16": Ux.size}.get(method, 0)
    return {"ACC": float(R[-1].mean()), "BWT": float(np.mean([R[-1, j] - R[j, j] for j in range(T - 1)])),
            "plasticity": float(np.diag(R).mean()), "rank_x": int(Ux.shape[1]), "rank_h": int(Uh.shape[1]),
            "stored_input_basis": int(stored), "R": R.tolist()}


def run():
    tasks = load()
    res = {"config": dict(H=H, epochs=EPOCHS, lr=LR, momentum=MOM, seeds=list(SEEDS), angles=ANGLES, ewc_grid=EWC_GRID)}
    out = {}
    for m in METHODS:
        if m == "ewc":
            best = None
            for lam in EWC_GRID:
                rows = [run_method(m, tasks, s, lam) for s in SEEDS]
                a = np.mean([r["ACC"] for r in rows])
                if best is None or a > best[0]:
                    best = (a, lam, rows)
            rows, extra = best[2], {"lambda": best[1]}
        else:
            rows, extra = [run_method(m, tasks, s) for s in SEEDS], {}
        out[m] = {"ACC": float(np.mean([r["ACC"] for r in rows])), "ACC_sd": float(np.std([r["ACC"] for r in rows])),
                  "BWT": float(np.mean([r["BWT"] for r in rows])), "plasticity": float(np.mean([r["plasticity"] for r in rows])),
                  "rank_x": float(np.mean([r["rank_x"] for r in rows])), "rank_h": float(np.mean([r["rank_h"] for r in rows])),
                  "stored_input_basis": float(np.mean([r["stored_input_basis"] for r in rows])),
                  "per_seed_ACC": [r["ACC"] for r in rows], "R_seed0": rows[0]["R"], **extra}
    res["results"] = out
    print("ROTATED DIGITS, domain-incremental, shared head (5 seeds)")
    print(f"  {'method':11s} {'ACC':>6s} {'±sd':>6s} {'BWT':>7s} {'plast':>6s} {'rank_x':>6s} {'rank_h':>6s} {'stored x-basis':>14s}")
    for m in METHODS:
        d = out[m]; tag = f"  (lambda {d['lambda']:g})" if "lambda" in d else ""
        print(f"  {m:11s} {d['ACC']:6.3f} {d['ACC_sd']:6.3f} {d['BWT']:+7.3f} {d['plasticity']:6.3f} "
              f"{d['rank_x']:6.1f} {d['rank_h']:6.1f} {d['stored_input_basis']:14.0f}{tag}")
    diff = out["pyramid2x2"]["ACC"] - out["gpm16"]["ACC"]
    seed_wins = sum(a >= b for a, b in zip(out["pyramid2x2"]["per_seed_ACC"], out["gpm16"]["per_seed_ACC"]))
    res["verdict"] = {"pyramid2x2_minus_gpm16": diff, "pyramid_wins_seeds": seed_wins, "pass": bool(diff >= 0)}
    print(f"\nVERDICT: pyramid2x2 - gpm16 = {diff:+.4f} (pyramid >= gpm16 on {seed_wins}/5 seeds) -> {'PASS' if diff >= 0 else 'FAIL'}")
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g6b_continual_rotated.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
