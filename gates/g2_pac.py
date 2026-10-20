"""G2 - Can a slow rhythm turn position into time (phase-gated release)?

Setup: macro-blocks bridged in a CHAIN (0-1-2-...-5). The soma (node 0, in block 0)
also injects a slow sinusoidal current, like a pacemaker. In steady state every node
oscillates with a phase psi_i that the medium itself sets; nothing is labelled.

Part A (measurement, the real question): does the medium give each macro-block its
own phase? Adjacent-block phase separation vs within-block phase spread, scanned over
drive frequency, for chain vs all-to-all topology.

Part B (use): K units in K distinct macro-blocks fire with unknown amplitude.
  linear: all release at t = 0, soma sums them (G1's situation).
  gated : each unit releases at the peak of its LOCAL slow oscillation,
          t_i = (-psi_i mod 2pi) / omega   (a PAC-like gate: fast burst at slow peak).
Both read the soma over the SAME window [0, T]. Decode the active set by NNLS over
per-block templates. Also with release-time jitter.

Honesty note written before running: the gate is an imposed nonlinearity, and
time-division multiplexing working once phases differ is close to guaranteed. What
is NOT guaranteed, and is the claim under test, is that a passive medium driven from
one point produces block-distinct phases at all, and that they are separated by more
than the slow smearing of the soma read-out.
"""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox import build, impulse, transfer, nnls_safe

M, S, n = 6, 4, 8
SEEDS = range(5)
OMEGAS = np.geomspace(0.002, 0.5, 25)
SOMA = 0
KS = [1, 2, 3, 4]
NOISE = 0.01
JITTER = [0.0, 2.0, 5.0]
TRIALS = 200


def circ_mean(a):
    return np.angle(np.exp(1j * a).mean())


def circ_sd(a):
    R = np.abs(np.exp(1j * a).mean())
    return np.sqrt(-2 * np.log(max(R, 1e-12)))


def phase_profile(med, omega):
    V = transfer(med, SOMA, omega)
    psi = np.angle(V)
    blocks = range(med.macro.max() + 1)
    means = np.array([circ_mean(psi[med.macro == b]) for b in blocks])
    within = np.array([circ_sd(psi[med.macro == b]) for b in blocks])
    return psi, means, within


def separation_score(means, within, chain=True):
    """min angular gap between ANY two block means, divided by the mean within-block SD."""
    d = np.abs(np.angle(np.exp(1j * (means[:, None] - means[None, :]))))
    np.fill_diagonal(d, np.inf)
    return float(d.min() / max(within.mean(), 1e-9)), float(d.min())


def run():
    res = {"config": dict(M=M, S=S, n=n, omegas=OMEGAS.tolist(), K=KS, noise=NOISE, jitter=JITTER), "seeds": []}
    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        row = {"seed": seed}
        # ---- Part A: phase map ----
        for topo in ["chain", "all"]:
            med = build(M, S, n, topology=topo, seed=seed)
            scores = []
            for om in OMEGAS:
                _, means, within = phase_profile(med, om)
                sc, gap = separation_score(means, within)
                scores.append((sc, gap, means.tolist(), within.tolist()))
            best = int(np.argmax([s[0] for s in scores]))
            row[topo] = {"best_omega": float(OMEGAS[best]), "best_score": scores[best][0],
                         "best_min_gap_deg": float(np.degrees(scores[best][1])),
                         "block_phase_deg": np.degrees(scores[best][2]).tolist(),
                         "within_sd_deg": np.degrees(scores[best][3]).tolist(),
                         "score_curve": [s[0] for s in scores]}
        # ---- Part B: use the chain medium at its best omega ----
        med = build(M, S, n, topology="chain", seed=seed)
        om = row["chain"]["best_omega"]
        psi, _, _ = phase_profile(med, om)
        period = 2 * np.pi / om
        T = period + 6.0 / med.lam[1:M].min()          # one cycle + slow tail
        ts = np.linspace(0.0, T, 1500)
        cand = np.where(med.sub != med.sub[SOMA])[0]
        blocks = np.arange(M)
        t_rel = (np.mod(-psi, 2 * np.pi)) / om

        dt = ts[1] - ts[0]
        Hbase = impulse(med, SOMA, ts)                 # all nodes, release at t=0

        def trace(node, t0):
            k = int(round(t0 / dt))
            out = np.zeros(len(ts))
            if k < len(ts):
                out[k:] = Hbase[node, : len(ts) - k]
            return out

        # templates per block (centroid over the block's candidate nodes)
        tmpl_lin = np.array([np.mean([trace(i, 0.0) for i in cand[med.macro[cand] == b]], 0) for b in blocks])
        tmpl_gat = np.array([np.mean([trace(i, t_rel[i]) for i in cand[med.macro[cand] == b]], 0) for b in blocks])
        scale = np.sqrt((tmpl_lin ** 2).mean())
        partB = {}
        for jit in JITTER:
            for K in KS:
                hits = {"linear": 0, "gated": 0}
                for _ in range(TRIALS):
                    chosen = rng.choice(blocks, K, replace=False)
                    amps = rng.uniform(0.5, 1.5, K)
                    nodes = [rng.choice(cand[med.macro[cand] == b]) for b in chosen]
                    y_lin = sum(a * trace(i, 0.0) for a, i in zip(amps, nodes))
                    y_gat = sum(a * trace(i, max(0.0, t_rel[i] + jit * rng.standard_normal()))
                                for a, i in zip(amps, nodes))
                    for name, y, D in [("linear", y_lin, tmpl_lin), ("gated", y_gat, tmpl_gat)]:
                        y = y + NOISE * scale * rng.standard_normal(len(ts))
                        coef = nnls_safe(D.T, y)
                        hits[name] += set(np.argsort(coef)[::-1][:K]) == set(chosen)
                partB[f"K{K}_jit{jit}"] = {k: v / TRIALS for k, v in hits.items()}
        row["partB"] = partB
        row["release_time_by_block"] = [float(np.median(t_rel[cand][med.macro[cand] == b])) for b in blocks]
        row["period"] = float(period)
        res["seeds"].append(row)
        c, a = row["chain"], row["all"]
        print(f"seed {seed}: CHAIN best w={c['best_omega']:.4f} score={c['best_score']:.1f} min gap={c['best_min_gap_deg']:.1f}deg "
              f"phases={np.round(c['block_phase_deg'], 0).tolist()} | ALL score={a['best_score']:.1f} gap={a['best_min_gap_deg']:.1f}deg")
        print("   " + "  ".join(f"K{K}: lin {partB[f'K{K}_jit0.0']['linear']:.2f} / gated {partB[f'K{K}_jit0.0']['gated']:.2f}"
                                for K in KS))
    summ = {}
    for topo in ["chain", "all"]:
        summ[f"{topo}:score"] = float(np.mean([r[topo]["best_score"] for r in res["seeds"]]))
        summ[f"{topo}:min_gap_deg"] = float(np.mean([r[topo]["best_min_gap_deg"] for r in res["seeds"]]))
    for jit in JITTER:
        for K in KS:
            for name in ["linear", "gated"]:
                summ[f"K{K}_jit{jit}:{name}"] = float(np.mean([r["partB"][f"K{K}_jit{jit}"][name] for r in res["seeds"]]))
    res["summary"] = summ
    print("\nmean over seeds")
    print(f"  phase separation score  chain {summ['chain:score']:.2f} (min gap {summ['chain:min_gap_deg']:.1f} deg)   "
          f"all-to-all {summ['all:score']:.2f} (min gap {summ['all:min_gap_deg']:.1f} deg)")
    for jit in JITTER:
        print(f"  jitter {jit}: " + "  ".join(
            f"K{K} lin {summ[f'K{K}_jit{jit}:linear']:.2f} gated {summ[f'K{K}_jit{jit}:gated']:.2f}" for K in KS))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g2_pac.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
