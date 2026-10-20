"""G4 - Does interface-total homeostasis rescue Sol's learned content from bridge drift?

Source: anttiluode/BrainBoxSol (branch sol/local-material-credit-g0, 973bb63). His G0
found that after local adjoint writes, -20/+20/-20% bridge drift raises read-out errors
to ~15%, and even 160 supervised replay steps with ALL old targets leave ~10-12%.
Gemini/Argon proposed that Claude's phase-gated addressing could protect the content.
G3b here says the address lives in INTERFACE TOTALS instead. This gate tests that on
Sol's own medium and writer (ported in brainbox/solport.py, reproduction checked below).

Sol's medium has ONE bridge edge per interface (P = 1). There the interface-total
homeostat is identical to restoring every bridge exactly (it would have to store each
weight), so P = 1 is a degenerate case and is NOT counted as evidence. The test uses
P parallel bridges per interface with the same total conductance, and drift that is
partly coherent per interface and partly independent per edge:
    g_e <- g_e * exp(s_c * xi_interface + s_i * xi_edge)
The homeostat stores ONE scalar per interface (its total) and gets no targets, no
replay and no read-out.

Repairs compared after drift (same drifted medium for all):
  none, replay (Sol's: 160 local supervised steps with all old targets),
  interface_total (homeostat), node_scaling (classic synaptic scaling, control),
  oracle (restore every bridge exactly = pre-drift error, upper bound).

PREDICTIONS, written before running:
  P1. P = 1 port reproduces Sol's numbers (checked first; the run aborts if not).
  P2. For P >= 2, interface_total removes most of the drift damage (> 70% on average)
      with no targets, and beats Sol's supervised replay.
  P3. What it cannot remove (the within-interface redistribution) grows with the
      bridge/intra conductance ratio (blocks are less well-mixed relative to bridges).
  P4. node_scaling helps little (< 30% of damage removed).
"""
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox.solport import (SolMedium, teacher_medium, train, repair_replay,
                              repair_interface_total, repair_node_scaling)

PATTERNS = list(itertools.product((-1, 1), repeat=4))
ORDERS = list(itertools.permutations(range(4)))
REPAIRS = ["none", "replay", "interface_total", "node_scaling", "oracle"]


def err(box, g, target, base):
    return float(np.mean(np.abs((box.values(g) - target) / base)))


def drift(box, g, s_c, s_i, rng):
    g = g.copy()
    m = box.owner == -1
    xi_if = rng.standard_normal(box.blocks - 1)
    f = np.exp(s_c * xi_if[box.iface[m]] + s_i * rng.standard_normal(m.sum()))
    g[m] *= f
    return g


def check_reproduction():
    box = SolMedium()
    base = box.values(box.initial)
    tgt = box.values(teacher_medium(box, (1, -1, 1, -1)))
    s = train(box, tgt, base, [0, 1, 2, 3], "local")
    mv = s.copy(); mv[box.owner == -1] *= np.array([.8, 1.2, .8])
    local = err(box, s, tgt, base)
    post = (box.values(mv) - tgt) / base
    rep = (box.values(repair_replay(box, mv, tgt, base)) - tgt) / base
    ok = (abs(local - 0.00155) < 1e-4 and np.allclose(np.round(post, 4), [0.0402, -0.1467, 0, -0.1457])
          and np.allclose(np.round(rep, 4), [0.0, -0.1165, -0.0001, -0.1037], atol=1e-4))
    print(f"P1 reproduction of Sol G0: local {local:.5f}, post-drift {np.round(post, 4)}, replay {np.round(rep, 4)} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("port does not reproduce BrainBoxSol; stopping")
    return {"local_mean_err": local, "post_drift": post.tolist(), "replay": rep.tolist()}


def case_block(P, bridge, writer, cases, draws, s_c, s_i, seed):
    rng = np.random.default_rng(seed)
    box = SolMedium(P=P, bridge=bridge)
    base = box.values(box.initial)
    rows = []
    for signs, order in cases:
        tgt = box.values(teacher_medium(box, signs))
        g = train(box, tgt, base, order, writer)
        pre = err(box, g, tgt, base)
        for _ in range(draws):
            gd = drift(box, g, s_c, s_i, rng)
            r = {"pre": pre}
            r["none"] = err(box, gd, tgt, base)
            r["replay"] = err(box, repair_replay(box, gd, tgt, base), tgt, base)
            r["interface_total"] = err(box, repair_interface_total(box, gd, g), tgt, base)
            r["node_scaling"] = err(box, repair_node_scaling(box, gd, g), tgt, base)
            r["oracle"] = pre
            rows.append(r)
    out = {k: float(np.mean([r[k] for r in rows])) for k in ["pre"] + REPAIRS}
    # fraction of drift damage removed, per row then averaged (rows with real damage)
    for k in REPAIRS[1:4]:
        fr = [(r["none"] - r[k]) / (r["none"] - r["pre"]) for r in rows if r["none"] - r["pre"] > 1e-4]
        out[f"removed_{k}"] = float(np.mean(fr)) if fr else float("nan")
    out["interface_beats_replay"] = float(np.mean([r["interface_total"] < r["replay"] for r in rows]))
    out["n"] = len(rows)
    return out


def fmt(tag, d):
    return (f"{tag:34s} pre {d['pre']:.4f} | none {d['none']:.4f}  replay {d['replay']:.4f}  "
            f"iface {d['interface_total']:.4f}  node {d['node_scaling']:.4f} | removed: replay "
            f"{d['removed_replay']:+.2f} iface {d['removed_interface_total']:+.2f} node {d['removed_node_scaling']:+.2f}"
            f" | iface<replay {d['interface_beats_replay']:.2f}")


def run():
    res = {"source": "anttiluode/BrainBoxSol@973bb63", "reproduction": check_reproduction()}
    all_cases = [(p, o) for p in PATTERNS for o in ORDERS]                    # 384
    sub_cases = [(p, o) for p in PATTERNS for o in ORDERS[::6]]               # 64

    print("\nA. Sol's bridge total 0.055, all 384 cases, local writer, 3 drift draws, s_c = s_i = 0.2")
    res["A"] = {}
    for P in [1, 2, 4, 8]:
        d = case_block(P, 0.055, "local", all_cases, 3, 0.2, 0.2, seed=10 + P)
        res["A"][P] = d
        print(fmt(f"P={P}" + (" (degenerate)" if P == 1 else ""), d))

    print("\nB. Purely independent per-edge drift (s_i = 0.3, no coherent part), 64 cases, local writer")
    res["B"] = {}
    for P in [2, 4, 8]:
        d = case_block(P, 0.055, "local", sub_cases, 3, 0.0, 0.3, seed=20 + P)
        res["B"][P] = d
        print(fmt(f"P={P}", d))

    print("\nC. Sol's protected (OGD/GPM) writer, 64 cases, s_c = s_i = 0.2")
    res["C"] = {}
    for P in [1, 4]:
        d = case_block(P, 0.055, "protected", sub_cases, 3, 0.2, 0.2, seed=30 + P)
        res["C"][P] = d
        print(fmt(f"P={P}" + (" (degenerate)" if P == 1 else ""), d))

    print("\nD. Bridge/intra ratio sweep (P = 4, 64 cases, local writer, s_c = s_i = 0.2)")
    res["D"] = {}
    for bridge in [0.0275, 0.055, 0.11, 0.22, 0.45]:
        d = case_block(4, bridge, "local", sub_cases, 3, 0.2, 0.2, seed=40)
        res["D"][bridge] = d
        print(fmt(f"bridge {bridge} (ratio {bridge / 0.9:.3f})", d))

    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g4_sol_bridge_rescue.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
