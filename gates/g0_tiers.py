"""G0 - Does block geometry give a free hierarchical address readable at one soma?

1. Eigen-tiers: does the spectrum split into groups whose sizes are set by block
   counts (1 global, M-1 macro, M(S-1) sub, rest local)?
2. Digits: are the slow eigenvectors constant inside blocks (an address digit)?
3. Read-out: put the SAME unit pulse (same local state) at different nodes. From the
   soma trace alone, held-out decode which macro-block and which sub-block it was in,
   using only the early window (fast) or only the late window (slow).
4. Control: a flat random graph with the same edge count and conductances.

Pass condition (pre-stated): slow-window decode of both macro and sub is well above
chance AND above the fast window, AND the flat control sits at chance.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox import build, impulse, tiers

M, S, n = 4, 4, 8
SEEDS = range(5)
NOISE = [0.02, 0.05, 0.1]
ts_fast = np.linspace(0.05, 2.0, 60)
ts_slow = np.geomspace(2.0, 2000.0, 60)
SOMA = 0


def digit_sharpness(med, lo, hi, labels, within=None):
    """between-group SD / within-group SD of eigenvector entries, averaged over modes."""
    out = []
    for k in range(lo, hi):
        v = med.phi[:, k]
        groups = np.unique(labels)
        means = np.array([v[labels == g].mean() for g in groups])
        wsd = np.mean([v[labels == g].std() for g in groups])
        if within is not None:  # compare sub-block means only inside each macro-block
            bsd = np.mean([means[[np.where(groups == g)[0][0] for g in np.unique(labels[within == m])]].std()
                           for m in np.unique(within)])
        else:
            bsd = means.std()
        out.append(bsd / max(wsd, 1e-12))
    return float(np.median(out))


def heldout_decode(H, labels, nodes_by_sub, noise, rng, reps=30):
    """Nearest-centroid; centroids from half the nodes of each sub-block, test on the
    other half; fresh additive white noise on every trace (train and test)."""
    sigma = noise * np.sqrt((H ** 2).mean())
    accs = []
    for _ in range(reps):
        tr, te = [], []
        for nodes in nodes_by_sub:
            p = rng.permutation(nodes)
            tr += list(p[: len(p) // 2]); te += list(p[len(p) // 2:])
        tr, te = np.array(tr), np.array(te)
        Xtr = H[tr] + sigma * rng.standard_normal((len(tr), H.shape[1]))
        Xte = H[te] + sigma * rng.standard_normal((len(te), H.shape[1]))
        cls = np.unique(labels[tr])
        C = np.array([Xtr[labels[tr] == c].mean(0) for c in cls])
        pred = cls[np.argmin(((Xte[:, None, :] - C[None]) ** 2).sum(-1), 1)]
        accs.append((pred == labels[te]).mean())
    return float(np.mean(accs))


def run():
    res = {"config": dict(M=M, S=S, n=n, seeds=list(SEEDS), noise=NOISE), "seeds": []}
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        row = {"seed": seed}
        for topo in ["all", "flat"]:
            med = build(M, S, n, topology=topo, seed=seed)
            tr = tiers(med.lam)
            sizes = [b - a for a, b in tr]
            keep = med.sub != med.sub[SOMA]          # exclude soma's own sub-block
            nodes = np.where(keep)[0]
            subs = [nodes[med.sub[nodes] == s] for s in np.unique(med.sub[nodes])]
            Hf = impulse(med, SOMA, ts_fast)
            Hs = impulse(med, SOMA, ts_slow)
            d = {"tier_sizes": sizes[:4], "lam_first": [float(x) for x in med.lam[:20]]}
            if topo == "all" and len(tr) >= 3:
                d["digit_macro"] = digit_sharpness(med, *tr[1], med.macro)
                d["digit_sub"] = digit_sharpness(med, *tr[2], med.sub, within=med.macro)
            for nz in NOISE:
                for wname, H in [("fast", Hf), ("slow", Hs)]:
                    d[f"{wname}_macro@{nz}"] = heldout_decode(H, med.macro, subs, nz, rng)
                    d[f"{wname}_sub@{nz}"] = heldout_decode(H, med.sub, subs, nz, rng)
            row[topo] = d
        res["seeds"].append(row)
        print(f"seed {seed}: tiers(all)={row['all']['tier_sizes']}  "
              f"digit macro={row['all'].get('digit_macro', float('nan')):.1f} sub={row['all'].get('digit_sub', float('nan')):.1f}  "
              f"| @0.05 all: fast macro {row['all']['fast_macro@0.05']:.2f} sub {row['all']['fast_sub@0.05']:.2f}  "
              f"slow macro {row['all']['slow_macro@0.05']:.2f} sub {row['all']['slow_sub@0.05']:.2f}  "
              f"| flat slow macro {row['flat']['slow_macro@0.05']:.2f} sub {row['flat']['slow_sub@0.05']:.2f}")
    # summary
    summ = {}
    for topo in ["all", "flat"]:
        for nz in NOISE:
            for w in ["fast", "slow"]:
                for lvl in ["macro", "sub"]:
                    k = f"{w}_{lvl}@{nz}"
                    v = [r[topo][k] for r in res["seeds"]]
                    summ[f"{topo}:{k}"] = [float(np.mean(v)), float(np.std(v))]
    summ["chance_macro"] = 1 / M
    summ["chance_sub"] = 1 / (M * S - 1)
    res["summary"] = summ
    ok = (summ["all:slow_macro@0.05"][0] > 2 * summ["chance_macro"]
          and summ["all:slow_sub@0.05"][0] > 4 * summ["chance_sub"]
          and summ["all:slow_macro@0.05"][0] > summ["all:fast_macro@0.05"][0]
          and summ["all:slow_sub@0.05"][0] > summ["all:fast_sub@0.05"][0]
          and summ["flat:slow_macro@0.05"][0] < 1.6 * summ["chance_macro"])
    res["verdict"] = "PASS" if ok else "FAIL"
    print("\nmean over seeds (noise 0.05):")
    for topo in ["all", "flat"]:
        print(f"  {topo:5s} fast macro {summ[f'{topo}:fast_macro@0.05'][0]:.2f}  sub {summ[f'{topo}:fast_sub@0.05'][0]:.2f}   "
              f"slow macro {summ[f'{topo}:slow_macro@0.05'][0]:.2f}  sub {summ[f'{topo}:slow_sub@0.05'][0]:.2f}")
    print(f"  chance macro {summ['chance_macro']:.2f}  sub {summ['chance_sub']:.3f}")
    print("VERDICT:", res["verdict"])
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g0_tiers.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
