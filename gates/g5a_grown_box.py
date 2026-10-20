"""G5a - The grown box: growth by splitting writes the address, but who can read it?

Growth rule: start with one box; every generation each box splits in two. Nodes that
were separated at generation g (0 = first, outermost split) are coupled with
conductance c * r^(L-1-g) / N: older splits are outer and weaker. Leaves = 2^L nodes.
This is an ultrametric medium: coupling depends only on the generation at which two
nodes were separated. A leaf's address is its split history (L binary digits).

Pre-stated expectations:
 E1. One tier of timescales per generation, tier sizes 1, 2, 4, ... and the tier
     eigenspaces are spanned by Haar functions (+ on one half, - on the other half of
     a box) of that generation: the medium's own modes ARE the block pyramid.
 E2. Slowest timescale grows as r^-(L-1): the outer box diverges in time.
 E3. (written before checking) Because coupling depends only on ultrametric distance,
     a single soma sees only the DISTANCE to a source (the generation of the first
     differing digit), not its full address: L+1 trace classes out of 2^L nodes.
     The full address needs symmetry breaking (imperfect growth) or several receivers.
 E4. Reading the distance digit for far (outer) sources needs an observation window
     of order the outer timescale; the readable depth is bounded by the window.
"""
import json, os, sys
import numpy as np
import scipy.linalg as sl

L_DEPTH = 6
RS = [0.1, 0.3]
SOMA = 0
OUT = {}


def grow(L, r, c=1.0, jitter=0.0, rng=None, leak=1e-6):
    N = 2 ** L
    i, j = np.triu_indices(N, 1)
    b = np.floor(np.log2(i ^ j)).astype(int)          # highest differing bit
    gen = L - 1 - b                                   # 0 = outermost split
    w = c * r ** (L - 1 - gen) / N
    if jitter > 0:
        w = w * np.exp(jitter * rng.standard_normal(len(w)))
    G = np.zeros((N, N)); G[i, j] = w; G[j, i] = w
    Lap = np.diag(G.sum(1)) - G + leak * np.eye(N)
    lam, phi = sl.eigh(Lap)
    return lam, phi


def haar_basis(L, g):
    """Haar functions of generation g: one per box existing at generation g."""
    N = 2 ** L
    size = N // 2 ** g
    H = []
    for start in range(0, N, size):
        v = np.zeros(N); v[start:start + size // 2] = 1; v[start + size // 2:start + size] = -1
        H.append(v / np.linalg.norm(v))
    return np.array(H).T


def traces(lam, phi, soma, ts):
    w = phi * phi[soma][None, :]
    return w @ np.exp(-np.outer(lam, ts))             # (N, T)


def distance_class(L, soma):
    N = 2 ** L
    d = np.zeros(N, int)
    for i in range(N):
        d[i] = 0 if i == soma else int(np.floor(np.log2(i ^ soma))) + 1   # 1..L
    return d


def run():
    rng = np.random.default_rng(7)
    L = L_DEPTH
    N = 2 ** L
    for r in RS:
        res = {}
        lam, phi = grow(L, r)
        mu = lam[1:]
        # E1 tiers + Haar overlap
        tiers = [[1]]
        for k in range(2, N):
            (tiers[-1].append(k) if lam[k] / lam[k - 1] < 1.5 else tiers.append([k]))
        sizes = [len(t) for t in tiers]
        overl = []
        for g, t in enumerate(tiers):
            P = phi[:, t]
            H = haar_basis(L, g)
            overl.append(float(np.linalg.norm(P.T @ H) ** 2 / max(len(t), H.shape[1])))
        res["tier_sizes"] = sizes
        res["haar_overlap"] = overl
        # E2 timescales
        res["slowest_tau"] = float(1 / mu[0]); res["fastest_tau"] = float(1 / mu[-1])
        # E3 degeneracy at one soma (noise-free, numerical classes)
        ts = np.geomspace(0.1, 50 * res["slowest_tau"], 200)
        Hs = traces(lam, phi, SOMA, ts)
        Hn = np.round(Hs / np.abs(Hs).max(), 9)
        res["distinct_trace_classes_one_soma"] = int(len({tuple(row) for row in Hn}))
        # receivers: how many distinct address signatures with R random receivers
        rec = {}
        for R in [1, 2, 4, 8, 16]:
            vals = []
            for _ in range(20):
                somas = rng.choice(N, R, replace=False)
                sig = np.concatenate([np.round(traces(lam, phi, s, ts) / np.abs(Hs).max(), 9) for s in somas], 1)
                vals.append(len({tuple(row) for row in sig}) / N)
            rec[R] = float(np.mean(vals))
        res["fraction_unique_addresses_vs_receivers"] = rec
        # E4 distance-digit decode vs observation window (1% noise), one soma
        dcls = distance_class(L, SOMA)
        win = {}
        for T in [1, 10, 100, 1e3, 1e4, 1e5, 1e6]:
            tw = np.geomspace(0.1, T, 120)
            Hw = traces(lam, phi, SOMA, tw)
            sigma = 0.01 * np.sqrt((Hw ** 2).mean())
            C = np.array([Hw[dcls == c].mean(0) for c in range(1, L + 1)])
            acc_by = np.zeros(L)
            for c in range(1, L + 1):
                idx = np.where(dcls == c)[0]
                hits = 0
                for _ in range(40):
                    i = rng.choice(idx)
                    x = Hw[i] + sigma * rng.standard_normal(len(tw))
                    hits += 1 + int(np.argmin(((C - x) ** 2).sum(1))) == c
                acc_by[c - 1] = hits / 40
            win[T] = acc_by.tolist()
        res["distance_decode_by_window"] = {str(k): v for k, v in win.items()}
        # symmetry breaking: imperfect growth, full address decode at one soma
        brk = {}
        for jit in [0.0, 0.05, 0.2, 0.5]:
            accs = {}
            lam2, phi2 = grow(L, r, jitter=jit, rng=rng)
            tw = np.geomspace(0.1, 50 / lam2[1], 200)
            Hw = traces(lam2, phi2, SOMA, tw)[1:]           # exclude soma itself
            for nz in [1e-6, 0.001, 0.01]:   # not 0: exact ties between identical traces are float artefacts
                sigma = nz * np.sqrt((Hw ** 2).mean())
                hits = 0
                for _ in range(300):
                    i = rng.integers(len(Hw))
                    x = Hw[i] + sigma * rng.standard_normal(len(tw))
                    hits += int(np.argmin(((Hw - x) ** 2).sum(1))) == i
                accs[str(nz)] = hits / 300
            brk[str(jit)] = accs
        res["full_address_one_soma_vs_jitter"] = brk
        OUT[str(r)] = res
        print(f"\nr = {r}  (depth {L}, {N} leaves)")
        print(f"  E1 tier sizes {sizes}   Haar overlap per tier {np.round(overl, 3).tolist()}")
        print(f"  E2 slowest tau {res['slowest_tau']:.3g}  fastest tau {res['fastest_tau']:.3g}")
        print(f"  E3 distinct trace classes at one soma: {res['distinct_trace_classes_one_soma']} (of {N} nodes)")
        print(f"     fraction of addresses unique vs receivers R: " + ", ".join(f"R{k}:{v:.2f}" for k, v in rec.items()))
        print(f"  E4 distance-digit decode (1% noise) per distance class 1..{L} (1 = sibling, {L} = farthest; "
              f"class {L} holds half of all nodes, so it wins near-zero ties at short windows):")
        for T, v in win.items():
            print(f"     window {T:>9.0f}: " + " ".join(f"{a:.2f}" for a in v))
        print(f"  full address at one soma vs growth jitter (noise 1e-6 / 0.001 / 0.01):")
        for j, a in brk.items():
            print(f"     jitter {j}: {a['1e-06']:.2f} / {a['0.001']:.2f} / {a['0.01']:.2f}   (chance {1/(N-1):.3f})")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g5a_grown_box.json"), "w") as f:
        json.dump(OUT, f, indent=1)


if __name__ == "__main__":
    run()
