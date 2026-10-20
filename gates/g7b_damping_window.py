"""G7b - the damping window G7a left open (0.001 churns, 0.003 freezes into a few domains).

His phiworld2 equations (brainbox/phimedium.py, m = None), 64x64 periodic, his random init, 3 seeds.
For each damping: number of domains (connected |phi| > 1 regions) at t = 800 and 1240, the fraction
of pixels keeping their sign after lags 4 / 40 / 440 (0.5 = unrelated), and the rms velocity.
Then: are the many domains of the hot state real? Quench a hot state (damping 0.0025, 48x72 Neumann)
with strong damping and count what survives.
"""
import json, os, sys
import numpy as np
from scipy.ndimage import label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox.phimedium import P0, step, evolve, noise_init

DAMPS = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.005]
SEEDS = [0, 1, 2]


def ndom(img, thr=1.0):
    return int(label(img > thr)[1] + label(img < -thr)[1])


def window():
    B = len(DAMPS) * len(SEEDS)
    p = dict(P0); p["damping"] = np.repeat(np.array(DAMPS), len(SEEDS))[:, None, None]
    phi, old = noise_init(np.random.default_rng(0), (B, 64, 64))
    dt = p["dt"]; snap = {}
    marks = {int(round(T / dt)): T for T in (800, 804, 840, 1240)}
    for k in range(1, max(marks) + 1):
        phi, old = step(phi, old, None, p, bc="periodic")
        if k in marks: snap[marks[k]] = phi.copy()
    rows = []
    for i in range(B):
        a = snap[800][i]; kept = []
        for L in (804, 840, 1240):
            b = snap[L][i]; mk = (abs(a) > 1) & (abs(b) > 1)
            kept.append(float((np.sign(a[mk]) == np.sign(b[mk])).mean()))
        r = dict(damping=float(p["damping"][i, 0, 0]), seed=i % len(SEEDS), domains_t800=ndom(a),
                 domains_t1240=ndom(snap[1240][i]), sign_kept_lag4_40_440=kept,
                 rms_velocity=float(np.sqrt(((snap[1240][i] - old[i]) ** 2).mean()) / dt))
        rows.append(r)
        print(f"damping {r['damping']:.4f} seed {r['seed']}: domains {r['domains_t800']:4d} -> {r['domains_t1240']:4d}   "
              f"sign kept after 4/40/440: {kept[0]:.2f}/{kept[1]:.2f}/{kept[2]:.2f}   rms velocity {r['rms_velocity']:.2f}")
    return rows


def quench():
    p = dict(P0, damping=0.0025)
    phi, old = noise_init(np.random.default_rng(0), (3, 48, 72))
    phi, old = evolve(phi, old, None, p, 800.0)
    out = dict(hot=[ndom(a) for a in phi])
    for qd in (0.05, 0.2, 0.5):
        f, o = evolve(phi.copy(), old.copy(), None, dict(P0, damping=qd), 400.0)
        out[f"quench_{qd}"] = [ndom(a) for a in f]
    print("hot state (damping 0.0025, t=800) domains:", out["hot"], " after quench:",
          {k: v for k, v in out.items() if k != "hot"})
    return out


if __name__ == "__main__":
    res = dict(window=window(), quench=quench())
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "g7b_damping_window.json"), "w"), indent=1)
