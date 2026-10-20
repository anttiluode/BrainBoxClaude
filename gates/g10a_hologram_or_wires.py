"""G10a - Is the learned m a hologram (fine fringes routing by diffraction) or coarse channels (refraction)?

Gemini (Oct 20 evening) read results/g8b_wires.png as a "volume Bragg grating" that routes crossing
channels by diffraction at K = k_in - k_out. That is testable on the saved media.

At the drive frequency (omega = 1.0) in the m0 = 0.7 domain the wave has k ~ 1.55 rad/cell
(wavelength ~ 4 cells): fringes written by forward x echo interference have spatial periods of ~2-4 cells.
Decompose each trained m into
  coarse = Gaussian low-pass (sigma = 2 cells: keeps structure longer than ~ 8-12 cells)
  fine   = m - coarse (the fringe band)
and rebuild media with the same total material:
  coarse only : m0 + (coarse - mean)          fine only : m0 + fine
  (clipped to [0, 1], then shifted to keep sum m)
Then re-relax phi and read the 3x3 transmission at omega.

Pre-stated:
  H (hologram): 'fine only' keeps the routing, 'coarse only' loses it; routing is narrow-band in omega
                (Bragg width ~ +-0.07 for a 31-cell crossing).
  W (wires / refraction): 'coarse only' keeps the routing, 'fine only' loses it.
Also reported: fraction of the variance of (m - mean) in the fringe band, and an omega sweep.
"""
import json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import g8_phiworld_wires as G
from brainbox.phimedium import P0

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MEDIA = [(f"G8b computed teacher, seed {s}", f"g8b_m_arm1_seed{s}.npy") for s in G.SEEDS] + \
        [(f"G9b physical teacher, seed {s}", f"g9b_m_seed{s}.npy") for s in G.SEEDS]
OMEGAS = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]


def same_total(m, total):
    for _ in range(50):
        m = np.clip(m + (total - m.sum()) / m.size, 0, 1)
    return m


def read(m, p, omega=1.0):
    G.OMEGA = omega
    v = np.sqrt(p["a"] * (1 - m) / p["b"])
    phi, old = G.relax(v, v.copy(), m, p, 200.0)
    T = G.fields(phi, old, m, p, echo=False)[2]
    G.OMEGA = 1.0
    return G.score(T, G.PI_CYCLIC), (np.abs(T) / G.AMP).round(4).tolist()


def run():
    p = dict(P0, form="variational", kappa=5.0)
    rows = []
    for name, f in MEDIA:
        m = np.load(os.path.join(OUT, f))[0].astype(float)
        total = m.sum()
        coarse = gaussian_filter(m, 2.0, mode="nearest")
        fine = m - coarse
        var_fine = float((fine ** 2).sum() / ((m - m.mean()) ** 2).sum())
        variants = {"trained": m,
                    "coarse only": same_total(G.M0 + (coarse - coarse.mean()), total),
                    "fine only": same_total(G.M0 + fine, total),
                    "trained, fine removed then x2 contrast": same_total(G.M0 + 2 * (coarse - coarse.mean()), total)}
        r = dict(medium=name, fine_band_variance_fraction=var_fine)
        for vn, mv in variants.items():
            s, T = read(mv, p)
            r[vn] = dict(acc=s["acc"], T_target=s["T_target"], logp=s["logp"], T=T)
        r["omega_sweep"] = {om: read(m, p, om)[0] for om in OMEGAS}
        rows.append(r)
        print(f"{name}: fine-band variance {var_fine:.2f} | " +
              " | ".join(f"{vn}: acc {r[vn]['acc']:.2f} T {r[vn]['T_target']:.3f}" for vn in variants) +
              " | omega sweep acc: " + " ".join(f"{om}:{r['omega_sweep'][om]['acc']:.2f}" for om in OMEGAS), flush=True)
    json.dump(rows, open(os.path.join(OUT, "g10a_hologram_or_wires.json"), "w"), indent=1)


if __name__ == "__main__":
    run()
