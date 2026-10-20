"""G6 - The AI question: does conserved-pyramid plasticity help continual learning?

Benchmark: split-digits, task-incremental. sklearn digits (8x8). Five tasks learned in
sequence: {0,1}, {2,3}, {4,5}, {6,7}, {8,9}. Network: x(64) -> W1 (64 hidden, tanh, no
bias, SHARED) -> one 2-way head per task. Forgetting happens through W1.
Plain SGD with momentum (Adam would break the projection guarantees of GPM / pyramid).

Methods (all train every task for the same number of epochs):
  finetune        : no protection
  ewc             : diagonal-Fisher EWC on W1, lambda picked from a grid by FINAL test
                    accuracy (optimistic for EWC; noted)
  gpm             : Gradient Projection Memory (Saha et al. 2021) on W1: after each task,
                    store an input-space basis capturing 95% of the not-yet-protected
                    energy; project later W1 gradients off it. Rank grows per task.
  gpm16 / gpm4    : GPM with the protected rank capped at 16 / 4 (matched to the pyramid)
  pyramid2x2      : after task 1, W1 updates must be zero-sum inside every local 2x2 pixel
                    block (block totals conserved). Protected rank 16. STORES NOTHING:
                    the subspace is the geometry.
  pyramid4x4      : same with 4x4 blocks. Protected rank 4.
  rand16          : zero-sum inside 16 random groups of 4 pixels (fixed, no locality).
Inputs: natural pixel order, and a fixed pixel permutation (locality destroyed).

PRE-STATED VERDICT RULE: the pyramid counts as an AI-relevant result only if, on natural
inputs, its final average accuracy (ACC) is >= gpm16's (equal protected rank) while storing
no basis. Beating finetune alone does not count. Expected (written before running): it
beats finetune and rand16, loses its edge on permuted inputs, and probably loses to gpm16
and gpm because a fixed geometric subspace cannot adapt to the data.
"""
import json, os
import numpy as np
from sklearn.datasets import load_digits

H = 64
EPOCHS = 150
LR = 0.1
MOM = 0.9
SEEDS = range(5)
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
EWC_GRID = [1.0, 10.0, 100.0, 1000.0]
METHODS = ["finetune", "ewc", "gpm", "gpm16", "gpm4", "pyramid2x2", "pyramid4x4", "rand16"]


def load(perm):
    d = load_digits()
    X = d.data / 16.0
    if perm is not None:
        X = X[:, perm]
    rng = np.random.default_rng(321)
    idx = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = idx[:cut], idx[cut:]
    out = []
    for a, b in TASKS:
        mtr = np.isin(d.target[tr], (a, b)); mte = np.isin(d.target[te], (a, b))
        out.append((X[tr][mtr], (d.target[tr][mtr] == b).astype(int),
                    X[te][mte], (d.target[te][mte] == b).astype(int)))
    return out


def group_basis(kind, rng):
    """Orthonormal basis (64 x r) of the subspace a zero-sum-in-groups update cannot touch:
    the span of the group indicator vectors."""
    grid = np.arange(64).reshape(8, 8)
    if kind == "2x2":
        G = [grid[i:i + 2, j:j + 2].ravel() for i in range(0, 8, 2) for j in range(0, 8, 2)]
    elif kind == "4x4":
        G = [grid[i:i + 4, j:j + 4].ravel() for i in (0, 4) for j in (0, 4)]
    elif kind == "rand":
        p = rng.permutation(64); G = [p[k:k + 4] for k in range(0, 64, 4)]
    B = np.zeros((64, len(G)))
    for c, g in enumerate(G):
        B[g, c] = 1 / np.sqrt(len(g))
    return B


def forward(W1, V, X):
    h = np.tanh(X @ W1.T)
    z = h @ V.T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return h, p


def grads(W1, V, X, y):
    h, p = forward(W1, V, X)
    dz = (p - np.eye(2)[y]) / len(X)
    gV = dz.T @ h
    dh = (dz @ V) * (1 - h ** 2)
    gW1 = dh.T @ X
    return gW1, gV


def acc(W1, V, X, y):
    return float((forward(W1, V, X)[1].argmax(1) == y).mean())


def run_method(method, tasks, seed, ewc_lam=0.0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((H, 64)) * np.sqrt(1 / 64)
    heads = [rng.standard_normal((2, H)) * np.sqrt(1 / H) for _ in tasks]
    U = np.zeros((64, 0))                     # protected input subspace (GPM / pyramid)
    fisher = np.zeros_like(W1); anchor = W1.copy()
    R = np.zeros((len(tasks), len(tasks)))   # R[i, j] = acc on task j after learning task i
    cap = {"gpm16": 16, "gpm4": 4}.get(method, None)
    for t, (Xtr, ytr, _, _) in enumerate(tasks):
        mW = np.zeros_like(W1); mV = np.zeros_like(heads[t])
        for _ in range(EPOCHS):
            gW, gV = grads(W1, heads[t], Xtr, ytr)
            if method == "ewc" and t > 0:
                gW = gW + ewc_lam * fisher * (W1 - anchor)
            if U.shape[1] > 0:
                gW = gW - (gW @ U) @ U.T     # remove the protected input directions
            mW = MOM * mW + gW; mV = MOM * mV + gV
            W1 -= LR * mW; heads[t] -= LR * mV
        for j in range(t + 1):
            R[t, j] = acc(W1, heads[j], tasks[j][2], tasks[j][3])
        # ---- consolidate after task t ----
        if method == "ewc":
            h, p = forward(W1, heads[t], Xtr)
            # empirical Fisher (diagonal) from per-sample gradients of log-likelihood
            dz = p - np.eye(2)[ytr]
            dh = (dz @ heads[t]) * (1 - h ** 2)          # (n, H)
            fisher += np.einsum("nh,nd->hd", dh ** 2, Xtr ** 2) / len(Xtr)
            anchor = W1.copy()
        elif method in ("gpm", "gpm16", "gpm4"):
            Xr = Xtr - (Xtr @ U) @ U.T                    # energy not yet protected
            total = (Xtr ** 2).sum()
            u, s, _ = np.linalg.svd(Xr.T, full_matrices=False)
            k = 0; acc_e = ((Xtr @ U) ** 2).sum()
            while k < len(s) and acc_e < 0.95 * total:
                acc_e += s[k] ** 2; k += 1
            if cap is not None:
                k = min(k, cap - U.shape[1])
            if k > 0:
                U = np.concatenate([U, u[:, :k]], 1)
        elif method.startswith("pyramid") or method == "rand16":
            if t == 0:
                kind = {"pyramid2x2": "2x2", "pyramid4x4": "4x4", "rand16": "rand"}[method]
                U = group_basis(kind, rng)
    final = R[-1]
    plast = np.diag(R)
    bwt = float(np.mean([R[-1, j] - R[j, j] for j in range(len(tasks) - 1)]))
    return {"ACC": float(final.mean()), "BWT": bwt, "plasticity": float(plast.mean()),
            "stored_numbers": int(U.size) if method.startswith("gpm") else (int(2 * W1.size) if method == "ewc" else 0),
            "protected_rank": int(U.shape[1]), "R": R.tolist()}


def run():
    rng = np.random.default_rng(77)
    perm = rng.permutation(64)
    res = {"config": dict(H=H, epochs=EPOCHS, lr=LR, momentum=MOM, seeds=list(SEEDS), tasks=TASKS, ewc_grid=EWC_GRID)}
    for order in ["natural", "permuted"]:
        tasks = load(None if order == "natural" else perm)
        out = {}
        for m in METHODS:
            if m == "ewc":
                best = None
                for lam in EWC_GRID:
                    rows = [run_method(m, tasks, s, lam) for s in SEEDS]
                    a = np.mean([r["ACC"] for r in rows])
                    if best is None or a > best[0]:
                        best = (a, lam, rows)
                rows = best[2]; extra = {"lambda": best[1]}
            else:
                rows = [run_method(m, tasks, s) for s in SEEDS]; extra = {}
            out[m] = {"ACC": float(np.mean([r["ACC"] for r in rows])), "ACC_sd": float(np.std([r["ACC"] for r in rows])),
                      "BWT": float(np.mean([r["BWT"] for r in rows])),
                      "plasticity": float(np.mean([r["plasticity"] for r in rows])),
                      "rank": float(np.mean([r["protected_rank"] for r in rows])),
                      "stored": float(np.mean([r["stored_numbers"] for r in rows])), **extra,
                      "per_seed_ACC": [r["ACC"] for r in rows]}
        res[order] = out
        print(f"\n{order.upper()} inputs   (5 seeds; ACC = final mean accuracy over 5 tasks; BWT = forgetting)")
        print(f"  {'method':12s} {'ACC':>6s} {'±sd':>6s} {'BWT':>7s} {'plast':>6s} {'rank':>5s} {'stored':>7s}")
        for m in METHODS:
            d = out[m]
            tag = f"  (lambda {d['lambda']:g})" if "lambda" in d else ""
            print(f"  {m:12s} {d['ACC']:6.3f} {d['ACC_sd']:6.3f} {d['BWT']:+7.3f} {d['plasticity']:6.3f} {d['rank']:5.1f} {d['stored']:7.0f}{tag}")
    nat = res["natural"]
    res["verdict"] = {
        "pyramid2x2_minus_gpm16_ACC": nat["pyramid2x2"]["ACC"] - nat["gpm16"]["ACC"],
        "pass": bool(nat["pyramid2x2"]["ACC"] >= nat["gpm16"]["ACC"]),
    }
    print(f"\nVERDICT (pre-stated rule): pyramid2x2 ACC - gpm16 ACC on natural inputs = "
          f"{res['verdict']['pyramid2x2_minus_gpm16_ACC']:+.4f} -> {'PASS' if res['verdict']['pass'] else 'FAIL'}")
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g6_continual.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
