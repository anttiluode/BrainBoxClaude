"""G3 - Two plasticities, two meanings? Content lives in blocks, address lives in bridges.

Medium: chain of 6 macro-blocks x 4 sub-blocks x 8 nodes (as in G2), 5 seeds.

CONTENT of sub-block b: its local fast fingerprint — pulse at the block's entry node,
voltage at all 8 of its own nodes over t in [0.05, 3]. Read LOCALLY.
ADDRESS of sub-block b: (i) its soma late-trace signature (G0 read-out), and
(ii) its macro-block's phase under the slow pacemaker drive (G2 read-out).

Drift: multiplicative log-normal noise exp(s * xi) on existing edges, applied to ONE
edge class at a time — "in" (inside sub-blocks) or "bridge" (between sub-blocks and
between macro-blocks). All read-outs after drift are decoded with STALE templates
taken before drift (a reader that has not re-learned anything).

Part A, pre-stated prediction — a double dissociation:
  in-block drift  -> content degrades, address barely moves
  bridge drift    -> address degrades, content barely moves

Part B (G3b), which conserved quantity protects the address under bridge drift:
  none                       : raw bridge drift
  interface-total homeostat  : rescale every bridge so each SUB-BLOCK-PAIR interface
                               keeps its total conductance (theory's prediction: the
                               address lives in the coarse operator = interface totals)
  macro-interface homeostat  : conserve only MACRO-pair totals (prediction: protects the
                               phase / macro address, not the sub-block address)
  per-node scaling (control) : classic synaptic scaling — each node restores its own
                               total conductance (prediction: fails, because bridges are
                               a small share of a node's total)
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox import build, impulse, transfer, conductances, from_conductances, response

M, S, n = 6, 4, 8
SOMA = 0
SEEDS = range(5)
DRIFT = [0.1, 0.3, 0.6, 1.0]
OMEGA = 0.01                      # inside G2b's good band
NOISE = 0.01
ts_loc = np.linspace(0.05, 3.0, 30)
ts_slow = np.geomspace(2.0, 5000.0, 60)


# ---------------------------------------------------------------- read-outs
def content_fingerprints(med):
    F = []
    for b in np.unique(med.sub):
        nodes = np.where(med.sub == b)[0]
        f = response(med, nodes[0], nodes, ts_loc).ravel()
        F.append(f / np.linalg.norm(f))
    return np.array(F)


def address_traces(med):
    return impulse(med, SOMA, ts_slow)            # (N, T)


def phases(med):
    return np.angle(transfer(med, SOMA, OMEGA))


class Reference:
    """Everything a stale reader learned before drift."""

    def __init__(self, med):
        self.macro, self.sub = med.macro, med.sub
        self.F = content_fingerprints(med)
        H = address_traces(med)
        self.keep = np.where(med.sub != med.sub[SOMA])[0]
        self.subs = np.unique(med.sub[self.keep])
        self.C = np.array([H[med.sub == s].mean(0) for s in self.subs])
        self.scale = np.sqrt((H[self.keep] ** 2).mean())
        psi = phases(med)
        self.block_phase = np.array([np.angle(np.exp(1j * psi[med.macro == m]).mean()) for m in range(M)])
        self.psi = psi

    def score(self, med, rng):
        out = {}
        # content: is each block's fingerprint still nearest its own old fingerprint?
        F = content_fingerprints(med)
        sim = F @ self.F.T
        out["content_id"] = float((sim.argmax(1) == np.arange(len(F))).mean())
        dself = np.linalg.norm(F - self.F, axis=1).mean()
        dpre = np.mean([np.linalg.norm(self.F[i] - self.F[j]) for i in range(len(F)) for j in range(len(F)) if i != j])
        out["content_drift_ratio"] = float(dself / dpre)
        # address (i): stale late-trace templates, sub-block level
        H = address_traces(med)[self.keep]
        X = H + NOISE * self.scale * rng.standard_normal(H.shape)
        pred = self.subs[np.argmin(((X[:, None, :] - self.C[None]) ** 2).sum(-1), 1)]
        out["address_sub_id"] = float((pred == self.sub[self.keep]).mean())
        Cnew = np.array([address_traces(med)[med.sub == s].mean(0) for s in self.subs])
        dself = np.linalg.norm(Cnew - self.C, axis=1).mean()
        dpre = np.mean([np.linalg.norm(self.C[i] - self.C[j]) for i in range(len(self.C)) for j in range(len(self.C)) if i != j])
        out["address_drift_ratio"] = float(dself / dpre)
        # address (ii): phase, macro level, stale block phases
        psi = phases(med)
        d = np.abs(np.angle(np.exp(1j * (psi[:, None] - self.block_phase[None, :]))))
        out["phase_macro_id"] = float((d.argmin(1) == self.macro).mean())
        out["phase_shift_deg"] = float(np.degrees(np.abs(np.angle(np.exp(1j * (psi - self.psi)))).mean()))
        return out


# ---------------------------------------------------------------- drift and homeostats
def edge_masks(med, G):
    E = G > 0
    same_sub = med.sub[:, None] == med.sub[None, :]
    return E & same_sub, E & ~same_sub


def drift(G, mask, s, rng):
    G = G.copy()
    iu = np.triu_indices_from(G, 1)
    m = mask[iu]
    f = np.exp(s * rng.standard_normal(m.sum()))
    vals = G[iu]
    vals[m] *= f
    G[iu] = vals
    G.T[iu] = vals
    return G


def interface_homeostat(G, G0, labels, bridge_mask):
    """Rescale bridges so every (label_a, label_b) interface keeps its original total."""
    G = G.copy()
    L = np.unique(labels)
    for a in L:
        for b in L:
            if b <= a:
                continue
            blk = bridge_mask & (labels[:, None] == a) & (labels[None, :] == b)
            blk = blk | blk.T
            tot0, tot = G0[blk].sum(), G[blk].sum()
            if tot > 0 and tot0 > 0:
                G[blk] *= tot0 / tot
    return G


def node_scaling(G, G0, iters=200):
    """Symmetric per-node synaptic scaling toward each node's original total conductance."""
    G = G.copy()
    target = G0.sum(1)
    for _ in range(iters):
        r = target / np.maximum(G.sum(1), 1e-12)
        G *= np.sqrt(np.outer(r, r))
    return G


# ---------------------------------------------------------------- run
def run():
    res = {"config": dict(M=M, S=S, n=n, drift=DRIFT, omega=OMEGA, noise=NOISE), "seeds": []}
    for seed in SEEDS:
        rng = np.random.default_rng(500 + seed)
        med0 = build(M, S, n, topology="chain", seed=seed)
        G0 = conductances(med0)
        ref = Reference(med0)
        in_m, br_m = edge_masks(med0, G0)
        row = {"seed": seed, "baseline": ref.score(med0, rng), "A": {}, "B": {}}
        for s in DRIFT:
            for cls, mask in [("in", in_m), ("bridge", br_m)]:
                G = drift(G0, mask, s, rng)
                row["A"][f"{cls}@{s}"] = ref.score(from_conductances(G, med0.macro, med0.sub), rng)
            # G3b on bridge drift
            Gb = drift(G0, br_m, s, rng)
            variants = {
                "none": Gb,
                "interface_total": interface_homeostat(Gb, G0, med0.sub, br_m),
                "macro_interface": interface_homeostat(Gb, G0, med0.macro, br_m & (med0.macro[:, None] != med0.macro[None, :])),
                "node_scaling": node_scaling(Gb, G0),
            }
            for k, G in variants.items():
                row["B"][f"{k}@{s}"] = ref.score(from_conductances(G, med0.macro, med0.sub), rng)
        res["seeds"].append(row)
        b = row["baseline"]
        print(f"seed {seed} baseline: content {b['content_id']:.2f} addr_sub {b['address_sub_id']:.2f} phase_macro {b['phase_macro_id']:.2f}")
    # summary
    keys = ["content_id", "content_drift_ratio", "address_sub_id", "address_drift_ratio", "phase_macro_id", "phase_shift_deg"]
    summ = {"baseline": {k: float(np.mean([r["baseline"][k] for r in res["seeds"]])) for k in keys}}
    for part in ["A", "B"]:
        for cond in res["seeds"][0][part]:
            summ[f"{part}:{cond}"] = {k: float(np.mean([r[part][cond][k] for r in res["seeds"]])) for k in keys}
    res["summary"] = summ
    hdr = f"{'condition':28s} {'content_id':>10s} {'cont_drift':>10s} {'addr_sub_id':>11s} {'addr_drift':>10s} {'phase_id':>9s} {'dphase':>7s}"

    def line(name, d):
        return (f"{name:28s} {d['content_id']:10.2f} {d['content_drift_ratio']:10.3f} {d['address_sub_id']:11.2f} "
                f"{d['address_drift_ratio']:10.3f} {d['phase_macro_id']:9.2f} {d['phase_shift_deg']:7.1f}")
    print("\nPART A — double dissociation (mean of seeds; stale templates)")
    print(hdr); print(line("baseline (no drift)", summ["baseline"]))
    for s in DRIFT:
        for cls in ["in", "bridge"]:
            print(line(f"{cls} drift s={s}", summ[f"A:{cls}@{s}"]))
    print("\nPART B — bridge drift + homeostats")
    print(hdr)
    for s in DRIFT:
        for k in ["none", "interface_total", "macro_interface", "node_scaling"]:
            print(line(f"{k} s={s}", summ[f"B:{k}@{s}"]))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g3_content_vs_address.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
