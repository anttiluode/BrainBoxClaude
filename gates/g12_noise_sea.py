"""G12 - the Noise Sea idea, measured: does noise make mass and slow the clock in PhiWorld?

His 2025 theory behind PhiWorld: under everything is noise; where noise concentrates it makes mass, and mass slows
time (c^2 = c0 / (1 + kappa phi^2)). Field theory has an exact version of the first half: in phi^4 theory,
fluctuations shift the mass (the 'thermal' or Hartree mass). Mean-field (Hartree) predictions for his potential
V'(phi) = -a phi + b phi^3, with fluctuation variance s2 = <dphi^2>:
  inside a domain (broken symmetry):  mean phi^2 -> a/b - 3 s2,  mass^2 = V'' -> 2a - 6 b s2   (noise LIGHTENS it)
  in the sea at phi = 0 (symmetric):  mass^2 = -a + 3 b s2                                 (noise CREATES mass)
  symmetry restored when s2 > a / (3b)  (= 1.67 for his a = 1, b = 0.2)
And for his clock, the noise-averaged c^2 = <1 / (1 + kappa phi^2)>: the function is convex for phi^2 > 1/(3 kappa),
concave near 0. So noise should SPEED the clock inside a domain and SLOW it in the sea around phi = 0.

Langevin PhiWorld (his equations, his update, 64x64 periodic): damping gamma = 0.05 and white force noise of strength
sigma, phi_new = phi + (1 - gamma dt)(phi - old) + dt^2 (accel + sigma xi / sqrt(dt)). Start: one + domain.
Measured after equilibration, for a ladder of sigma:
  mean phi, s2 = variance of phi, <c^2>, the k = 0 mode's frequency (from the spectrum of the spatial mean),
  and the fraction of pixels still positive.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox.phimedium import P0, accel

SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
N, GAMMA = 64, 0.05
T_EQ, T_MEAS = 400.0, 800.0


def run(kappa=5.0, seed=0):
    p = dict(P0, kappa=kappa)
    dt = p["dt"]
    B = len(SIGMAS)
    sig = np.array(SIGMAS)[:, None, None]
    rng = np.random.default_rng(seed)
    phi = np.full((B, N, N), np.sqrt(p["a"] / p["b"])) + 0.01 * rng.standard_normal((B, N, N))
    old = phi.copy()
    n_eq, n_meas = int(T_EQ / dt), int(T_MEAS / dt)
    means = []
    acc = dict(s1=0, s2=0, c2=0, pos=0)
    for k in range(n_eq + n_meas):
        f = sig * rng.standard_normal(phi.shape) / np.sqrt(dt)
        new = phi + (1 - GAMMA * dt) * (phi - old) + dt ** 2 * (accel(phi, None, p, bc="periodic") + f)
        old, phi = phi, new
        if k >= n_eq:
            means.append(phi.mean((1, 2)))
            acc["s1"] += phi.mean((1, 2)); acc["s2"] += (phi ** 2).mean((1, 2))
            acc["c2"] += (p["c0"] / (1 + kappa * phi ** 2)).mean((1, 2)); acc["pos"] += (phi > 0).mean((1, 2))
    mean = acc["s1"] / n_meas
    var = acc["s2"] / n_meas - mean ** 2
    means = np.array(means)                                   # (time, B)
    freqs = np.fft.rfftfreq(len(means), dt) * 2 * np.pi
    rows = []
    for i, s in enumerate(SIGMAS):
        x = means[:, i] - means[:, i].mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        lo = freqs > 0.2                                      # skip the slow drift
        w0 = float(freqs[lo][np.argmax(spec[lo])]) if s > 0 else float("nan")
        r = dict(sigma=s, mean_phi=float(mean[i]), var_phi=float(var[i]), mean_c2=float(acc["c2"][i] / n_meas),
                 frac_positive=float(acc["pos"][i] / n_meas), k0_frequency=w0,
                 hartree_mass_broken=float(np.sqrt(max(2 * p["a"] - 6 * p["b"] * var[i], 0))),
                 hartree_mass_sea=float(np.sqrt(max(-p["a"] + 3 * p["b"] * var[i], 0))))
        rows.append(r)
        print(f"kappa {kappa} sigma {s:5.2f}: mean phi {r['mean_phi']:+.2f}  var {r['var_phi']:.3f}  positive {r['frac_positive']:.2f}  "
              f"<c^2> {r['mean_c2']:.4f}  k=0 frequency {w0:.3f}  (Hartree: domain {r['hartree_mass_broken']:.3f}, "
              f"sea {r['hartree_mass_sea']:.3f})", flush=True)
    return rows


if __name__ == "__main__":
    res = {f"kappa_{k}": run(k) for k in (5.0, 0.0)}
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "g12_noise_sea.json"), "w"), indent=1)
