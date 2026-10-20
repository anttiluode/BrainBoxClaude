"""G8a - Are PhiWorld walls wires? Is an m-stripe a wire? (physics before any learning)

Same box as G8 (32 x 48, absorbing edges, damping 0.005, drive amplitude 0.02, lock-in read-out).
Drive the middle input port (x = 8) at omega; report the lock-in amplitude at the middle output
(x = 39, 31 cells away) and at the two neighbouring outputs, divided by the drive amplitude.

Media (m couples through the gap, a -> a (1 - m)):
  domain m=0      one uniform domain, full gap sqrt(2) (his parameters)
  domain m=0.7    one uniform domain, gap sqrt(0.6) = 0.77 (the G8 starting medium)
  wall            + domain above, - domain below, the wall on the port line, m = 0
  stripe w        m = 1 on a stripe of width w along the port line, m = 0 elsewhere
Each with kappa = 5 (his rule) and kappa = 0 (no mass-slows-the-clock).

Pre-stated: P1 the m=0 domain transmits ~nothing below omega = 1.41; P2 a wall transmits;
P3 an m=1 stripe transmits and leaks little to the neighbouring outputs.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox.phimedium import P0
import g8_phiworld_wires as G

OMEGAS = [0.6, 0.8, 1.0, 1.2]


def media():
    y, x = G._y, G._x
    line = np.abs(y - 15.5)
    span = (x >= G.X_IN - 2) & (x <= G.X_OUT + 2)
    zero = np.zeros((G.NY, G.NX))
    out = {"domain m=0": (None, zero), "domain m=0.7": (None, zero + 0.7), "wall": ("wall", zero)}
    for w in (2, 4, 6):
        m = zero.copy(); m[(line < w / 2) & span] = 1.0
        out[f"stripe w{w}"] = (None, m)
    return out


def run():
    rows = []
    for kappa in (5.0, 0.0):
        p = dict(P0, kappa=kappa)
        for name, (kind, m) in media().items():
            v = np.sqrt(p["a"] * (1 - m) / p["b"])
            if kind == "wall":
                v = np.where(G._y < 15.5, v, -v)
            phi, old = G.relax(v, v.copy(), m, p, 200.0)
            r = dict(kappa=kappa, medium=name, phi_on_line=float(phi[15:17, 24].mean()))
            for om in OMEGAS:
                G.OMEGA = om
                T = np.abs(G.fields(phi, old, m, p, echo=False)[2][1]) / G.AMP
                r[f"om{om}"] = dict(target=float(T[1]), neighbours=float((T[0] + T[2]) / 2))
            rows.append(r)
            print(f"kappa {kappa} {name:13s} phi(line) {r['phi_on_line']:+.2f}  " +
                  "  ".join(f"w={om}: {r[f'om{om}']['target']:.4f} (leak {r[f'om{om}']['neighbours']:.4f})" for om in OMEGAS))
    # his-rule wall: profile across it and a wider frequency sweep
    p = dict(P0)
    zero = np.zeros((G.NY, G.NX)); v = np.where(G._y < 15.5, np.sqrt(5.0), -np.sqrt(5.0)) + zero
    phi, old = G.relax(v, v.copy(), zero, p, 200.0)
    sweep = {}
    for om in (0.2, 0.4, 0.9, 1.1, 1.3, 1.38, 1.5, 1.7):
        G.OMEGA = om
        sweep[om] = float(np.abs(G.fields(phi, old, zero, p, echo=False)[2][1, 1]) / G.AMP)
    extra = dict(wall_profile_kappa5=phi[12:20, 24].round(3).tolist(), wall_sweep_kappa5=sweep)
    print("kappa 5 wall profile across (rows 12-19):", extra["wall_profile_kappa5"])
    print("kappa 5 wall, wider sweep:", {k: f"{v:.1e}" for k, v in sweep.items()})
    rows.append(extra)
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..", "results", "g8a_wires.json"), "w"), indent=1)


if __name__ == "__main__":
    run()
