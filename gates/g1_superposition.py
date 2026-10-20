"""G1 - What does a linear soma lose when several units are active at once?

Prediction written before running (expected to FAIL as a scaling claim):
  A single soma separates addresses by decay RATE in time. Modes with near-equal
  eigenvalues are temporally indistinguishable, so the number of address dimensions a
  soma can resolve is set by the number of distinct timescales (~ number of tiers),
  not by the number of slow modes (16 here). With K simultaneous units of unknown
  amplitude, exact recovery of the active set should collapse quickly with K.
  Adding receivers (more somas = more boundary points) should restore it.

(a) Resolvable dimension: singular values of the sub-block address dictionary
    (soma traces) vs a noise floor, for R = 1, 2, 4, 8 receivers.
(b) K units in K distinct sub-blocks, amplitudes U(0.5, 1.5). Decode the active set
    by non-negative least squares on the sub-block dictionary (K given to the decoder,
    which is generous). Exact-set recovery vs K, noise, R.
"""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox import build, impulse, nnls_safe

M, S, n = 4, 4, 8
SEEDS = range(5)
ts = np.concatenate([np.linspace(0.05, 2.0, 60), np.geomspace(2.0, 2000.0, 60)])
KS = [1, 2, 3, 4, 5]
NOISE = [0.0, 0.01, 0.05]
RS = [1, 2, 4, 8]
TRIALS = 200


def receivers(med, R, rng):
    """R somas: node 0 plus nodes in different macro-blocks (spread boundary points)."""
    out = [0]
    order = [m for m in range(1, med.macro.max() + 1)] * 4
    for m in order:
        if len(out) >= R:
            break
        cand = np.where((med.macro == m) & (~np.isin(med.sub, med.sub[out])))[0]
        out.append(int(rng.choice(cand)))
    return out[:R]


def traces(med, somas, sources):
    return np.concatenate([impulse(med, s, ts, sources) for s in somas], axis=1)


def run():
    res = {"config": dict(M=M, S=S, n=n, KS=KS, NOISE=NOISE, RS=RS, trials=TRIALS), "seeds": []}
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        med = build(M, S, n, topology="all", seed=seed)
        row = {"seed": seed}
        # Nested receiver sets and ONE fixed candidate list for every R, so that adding
        # receivers never shrinks the decoding problem (matched atom count).
        all_somas = receivers(med, max(RS), rng)
        cand_subs = [s for s in np.unique(med.sub) if s not in set(med.sub[all_somas])]
        for R in RS:
            somas = all_somas[:R]
            Hall = traces(med, somas, np.arange(med.N))
            D = np.array([Hall[med.sub == s].mean(0) for s in cand_subs])   # dictionary
            scale = np.sqrt((Hall[np.isin(med.sub, cand_subs)] ** 2).mean())
            sv = np.linalg.svd(D / scale, compute_uv=False)
            Tlen = D.shape[1]
            dims = {str(nz): int((sv > nz * np.sqrt(Tlen) * 3).sum()) if nz > 0 else int((sv > 1e-6 * sv[0]).sum())
                    for nz in NOISE}
            rec = {}
            for nz in NOISE:
                for K in KS:
                    hit = 0
                    for _ in range(TRIALS):
                        chosen = rng.choice(len(cand_subs), K, replace=False)
                        amps = rng.uniform(0.5, 1.5, K)
                        y = np.zeros(Tlen)
                        for c, a in zip(chosen, amps):
                            node = rng.choice(np.where(med.sub == cand_subs[c])[0])
                            y += a * Hall[node]
                        y += nz * scale * rng.standard_normal(Tlen)
                        coef = nnls_safe(D.T, y)
                        pred = np.argsort(coef)[::-1][:K]
                        hit += set(pred) == set(chosen)
                    rec[f"K{K}@{nz}"] = hit / TRIALS
            row[f"R{R}"] = {"somas": [int(s) for s in somas], "n_atoms": len(cand_subs),
                            "resolvable_dims": dims, "sv_head": [float(x) for x in (sv / sv[0])[:16]],
                            "recovery": rec}
            print(f"seed {seed} R={R} atoms={len(cand_subs)} dims@0.01={dims['0.01']} dims@0.05={dims['0.05']}  "
                  + "  ".join(f"K{K}:{rec[f'K{K}@0.01']:.2f}" for K in KS) + "  (noise 0.01)")
        res["seeds"].append(row)
    summ = {}
    for R in RS:
        for nz in NOISE:
            summ[f"R{R}:dims@{nz}"] = float(np.mean([r[f"R{R}"]["resolvable_dims"][str(nz)] for r in res["seeds"]]))
            for K in KS:
                summ[f"R{R}:K{K}@{nz}"] = float(np.mean([r[f"R{R}"]["recovery"][f"K{K}@{nz}"] for r in res["seeds"]]))
    res["summary"] = summ
    print("\nmean over seeds — exact active-set recovery")
    for nz in NOISE:
        print(f" noise {nz}")
        for R in RS:
            print(f"   R={R}: dims={summ[f'R{R}:dims@{nz}']:.1f}  " +
                  "  ".join(f"K{K} {summ[f'R{R}:K{K}@{nz}']:.2f}" for K in KS))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g1_superposition.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
