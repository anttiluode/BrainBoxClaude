"""G2b - Price of the phase gate: correctly addressed units per unit time.

G2 showed a chain medium turns position into slow phase, and phase-gated release then
recovers K=4 simultaneous units perfectly where a linear soma gets ~54%. But the best
drive period was ~1000 time units while a linear read is over in ~100. Time-division
multiplexing buys accuracy with time. This gate asks whether it buys THROUGHPUT.

For drive frequency omega on a grid:
  gated : one cycle of length P = 2pi/omega + slow tail; K units fire at their local
          peaks; throughput = K * P(exact set) / window.
  linear: back-to-back reads of length W = ln(100)/lambda_min_macro (slowest address
          mode decays to 1%); throughput = K * P(exact set) / W, best K.
Jitter on release time is given in absolute time units (biological spike timing
jitter does not scale with the rhythm).
Pre-stated expectation: gated wins on throughput only if the phase gap per block,
converted to time (gap/omega), stays larger than the soma's slow smear (~1/lambda_macro)
at a period short enough to beat W.
"""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox import build, impulse, transfer, nnls_safe

M, S, n = 6, 4, 8
SOMA = 0
SEEDS = range(3)
OMEGAS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
KS = [1, 2, 3, 4, 5]   # K=M would be trivially correct (one possible set)
JITTER = [0.0, 5.0, 15.0]
NOISE = 0.01
TRIALS = 150


def run():
    out = {"config": dict(M=M, omegas=OMEGAS, KS=KS, jitter=JITTER, noise=NOISE), "seeds": []}
    for seed in SEEDS:
        rng = np.random.default_rng(400 + seed)
        med = build(M, S, n, topology="chain", seed=seed)
        lam_min = med.lam[1:M].min()
        smear = 1.0 / lam_min
        cand = np.where(med.sub != med.sub[SOMA])[0]
        blocks = np.arange(M)
        row = {"seed": seed, "smear": float(smear)}

        def setup(T, npts=2000):
            ts = np.linspace(0, T, npts)
            return ts, ts[1] - ts[0], impulse(med, SOMA, ts)

        def shifted(Hb, dt, node, t0, L):
            k = int(round(t0 / dt)); o = np.zeros(L)
            if k < L:
                o[k:] = Hb[node, :L - k]
            return o

        def trial_acc(Hb, dt, L, trel, K, jit, gated):
            D = np.array([np.mean([shifted(Hb, dt, i, trel[i] if gated else 0.0, L)
                                   for i in cand[med.macro[cand] == b]], 0) for b in blocks])
            scale = np.sqrt((D ** 2).mean())
            hit = 0
            for _ in range(TRIALS):
                ch = rng.choice(blocks, K, replace=False)
                y = np.zeros(L)
                for b in ch:
                    i = rng.choice(cand[med.macro[cand] == b])
                    t0 = max(0.0, trel[i] + jit * rng.standard_normal()) if gated else 0.0
                    y += rng.uniform(0.5, 1.5) * shifted(Hb, dt, i, t0, L)
                y += NOISE * scale * rng.standard_normal(L)
                c = nnls_safe(D.T, y)
                hit += set(np.argsort(c)[::-1][:K]) == set(ch)
            return hit / TRIALS

        # linear reference
        W = np.log(100) / lam_min
        ts, dt, Hb = setup(W)
        lin = {K: trial_acc(Hb, dt, len(ts), np.zeros(med.N), K, 0.0, False) for K in KS}
        lin_tp = max(K * lin[K] / W for K in KS)
        row["linear"] = {"window": float(W), "acc": lin, "throughput": float(lin_tp)}
        row["gated"] = {}
        for om in OMEGAS:
            psi = np.angle(transfer(med, SOMA, om))
            trel = np.mod(-psi, 2 * np.pi) / om
            means = [np.angle(np.exp(1j * psi[med.macro == b]).mean()) for b in blocks]
            gaps = np.abs(np.angle(np.exp(1j * (np.array(means)[:, None] - np.array(means)[None]))))
            np.fill_diagonal(gaps, np.inf)
            dt_gap = gaps.min() / om
            T = 2 * np.pi / om + np.log(100) / lam_min
            ts, dt, Hb = setup(T)
            g = {}
            for jit in JITTER:
                acc = {K: trial_acc(Hb, dt, len(ts), trel, K, jit, True) for K in KS}
                P = 2 * np.pi / om
                g[str(jit)] = {"acc": acc,
                               "throughput": float(max(K * acc[K] / T for K in KS)),
                               "throughput_pipelined": float(max(K * acc[K] / P for K in KS))}
            row["gated"][str(om)] = {"window": float(T), "min_gap_deg": float(np.degrees(gaps.min())),
                                     "min_gap_time": float(dt_gap), "by_jitter": g}
            print(f"seed {seed} w={om:<6} window={T:7.1f} gap={np.degrees(gaps.min()):5.1f}deg = {dt_gap:6.1f}t "
                  f"(smear {smear:.1f})  " + "  ".join(
                      f"jit{j}: K4 {g[str(j)]['acc'][4]:.2f} tp {g[str(j)]['throughput']:.4f}/{g[str(j)]['throughput_pipelined']:.4f}" for j in JITTER)
                  + f"   | linear tp {lin_tp:.4f} (W={W:.0f}, K4 {lin[4]:.2f})")
        out["seeds"].append(row)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g2b_throughput.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    run()
