"""G7a - Does PhiWorld grow its own boxes? Spectrum of the frozen domain state.

PhiWorld2 (his 2025 code, equations copied exactly; GUI removed):
    phi_tt = c2(phi) * Lap(phi) - V'(phi) - gamma * Lap^2(phi)
    c2(phi) = c0 / (1 + kappa * phi^2)          (mass slows the clock)
    V'(phi) = -a*phi + b*phi^3                  (Mexican hat; minima at +-sqrt(a/b) = +-sqrt(5))
    defaults: c0 = 1, kappa = 5, a = 1, b = 0.2, gamma = 0.02, dt = 0.08, periodic grid.

Protocol: start from his 'random' init (uniform noise, amplitude 0.05), evolve with his
leapfrog update but with strong damping so the field settles into a (nearly) stationary
domain pattern; check stationarity; linearize around it:
    J = diag(c2) Lap + diag(c2'(phi) * Lap(phi)) - diag(V''(phi)) - gamma Lap^2
Small oscillations obey d2/dt2 delta = J delta, so an eigenvalue mu < 0 is a mode of angular
frequency sqrt(-mu); small |mu| = slow.

Pre-stated expectations:
 E1. The spectrum splits into a SOFT band (modes near zero: wall displacements, the
     Goldstone-like translation modes of phi^4 kinks) and a GAPPED band starting near
     sqrt(V''(+-sqrt5)) = sqrt(2) (domain interiors). Two tiers, separated by a gap.
 E2. Slow (soft) modes live on the WALLS (|phi| small), not inside domains — so in time,
     PhiWorld's domain interiors are the stiff part, not the slow part. "Mass slows the
     clock" slows PROPAGATION inside domains, not their oscillation frequency.
 E3. kappa = 0 control: same domains, but propagation inside domains is ~26x faster
     (c2 = 1 vs 1/26); the soft band should persist (it comes from the walls).

REVISED after the first run (kept in the ledger): the frozen states had coarsened to flat
stripes (walls 2-4% of area), and the largest-gap tier detector was too crude. Analysis now
counts modes below the uniform-domain band edge sqrt(2a) = sqrt(2) (they can only be bound
to walls) and their wall enrichment; and adds the question that matters for persistent
"boxes": does mass-slows-the-clock (kappa) slow down domain coarsening?
"""
import json, os
import numpy as np
import scipy.sparse as sp

N = 48
P = dict(c0=1.0, kappa=5.0, a=1.0, b=0.2, gamma=0.02, dt=0.08)
RELAX_DAMPING = 2.0
STEPS = 30000
SEEDS = [0, 1, 2]


def lap_matrix(n):
    I = sp.identity(n, format="csr")
    D = sp.diags([1., 1., -2., 1., 1.], [-(n - 1), -1, 0, 1, n - 1], shape=(n, n), format="csr")
    return (sp.kron(D, I) + sp.kron(I, D)).tocsr()          # periodic 5-point Laplacian


def accel(phi, Lp, p):
    lap = Lp @ phi
    c2 = p["c0"] / (1 + p["kappa"] * phi ** 2)
    return c2 * lap - (-p["a"] * phi + p["b"] * phi ** 3) - p["gamma"] * (Lp @ lap)


def relax(seed, p, Lp):
    rng = np.random.default_rng(seed)
    phi = (rng.random(N * N) - 0.5) * 0.1
    old = phi.copy()
    dt = p["dt"]
    for k in range(STEPS):
        new = phi + (1 - RELAX_DAMPING * dt) * (phi - old) + dt ** 2 * accel(phi, Lp, p)
        old, phi = phi, new
    vel = np.linalg.norm(phi - old) / dt / np.sqrt(N * N)
    res = np.linalg.norm(accel(phi, Lp, p)) / np.sqrt(N * N)
    return phi, vel, res


def jacobian(phi, Lp, p):
    L = Lp.toarray()
    lap = Lp @ phi
    c2 = p["c0"] / (1 + p["kappa"] * phi ** 2)
    dc2 = -p["c0"] * 2 * p["kappa"] * phi / (1 + p["kappa"] * phi ** 2) ** 2
    Vpp = -p["a"] + 3 * p["b"] * phi ** 2
    return np.diag(c2) @ L + np.diag(dc2 * lap - Vpp) - p["gamma"] * (L @ L)


def analyse(phi, p, Lp):
    J = jacobian(phi, Lp, p)
    mu, vec = np.linalg.eig(J)
    order = np.argsort(-mu.real)                              # least negative first (slowest)
    mu, vec = mu[order], vec[:, order]
    wall = np.abs(phi) < 1.0
    dom = np.abs(phi) > 0.8 * np.sqrt(p["a"] / p["b"])
    w = np.abs(vec) ** 2; w /= w.sum(0, keepdims=True)
    on_wall = w[wall].sum(0)
    freq = np.sqrt(np.clip(-mu.real, 0, None))
    unstable = int((mu.real > 1e-6).sum())
    bulk_gap = np.sqrt(2 * p["a"])          # sqrt(V''(+-sqrt(a/b))) = sqrt(2a): uniform-domain band edge
    sub = freq < 0.98 * bulk_gap
    area = wall.mean()
    return {
        "wall_fraction_of_area": float(area), "domain_fraction_of_area": float(dom.mean()),
        "unstable_modes": unstable, "max_imag_over_real": float(np.max(np.abs(mu.imag)) / np.max(np.abs(mu.real))),
        "bulk_gap": float(bulk_gap), "n_subgap": int(sub.sum()),
        "subgap_freqs": sorted(freq[sub].tolist())[:40],
        "subgap_wall_weight": float(on_wall[sub].mean()) if sub.any() else float("nan"),
        "abovegap_wall_weight": float(on_wall[~sub].mean()),
        "subgap_wall_enrichment": float(on_wall[sub].mean() / area) if sub.any() and area > 0 else float("nan"),
        "slowest_10_freq": sorted(freq.tolist())[:10],
    }


def coarsening(kappa, damping, seed, n=64, steps=20000, every=500):
    """Domain count over time from noise, at a given damping (his GUI default is 0.001)."""
    from scipy.ndimage import label
    p = dict(P, kappa=kappa)
    Lp = lap_matrix(n)
    rng = np.random.default_rng(seed)
    phi = (rng.random(n * n) - 0.5) * 0.1; old = phi.copy(); dt = p["dt"]
    counts = []
    for k in range(1, steps + 1):
        new = phi + (1 - damping * dt) * (phi - old) + dt ** 2 * accel(phi, Lp, p)
        old, phi = phi, new
        if not np.isfinite(phi).all():
            counts.append(("diverged", k)); break
        if k % every == 0:
            img = phi.reshape(n, n)
            npos = label(img > 1.0)[1]; nneg = label(img < -1.0)[1]
            counts.append((k * dt, int(npos + nneg), float(np.sqrt((phi - old) @ (phi - old) / n / n) / dt)))
    return counts


def run():
    Lp = lap_matrix(N)
    res = {"config": dict(N=N, params=P, relax_damping=RELAX_DAMPING, steps=STEPS, seeds=SEEDS)}
    for kappa in [5.0, 0.0]:
        p = dict(P, kappa=kappa)
        rows = []
        for s in SEEDS:
            phi, vel, r = relax(s, p, Lp)
            a = analyse(phi, p, Lp)
            a.update(seed=s, rms_velocity=float(vel), residual_accel=float(r),
                     phi_range=[float(phi.min()), float(phi.max())])
            rows.append(a)
            print(f"kappa={kappa} seed {s}: |v|={vel:.1e} |A|={r:.1e}  phi in [{phi.min():.2f},{phi.max():.2f}]  "
                  f"walls {a['wall_fraction_of_area']:.3f} of area  unstable {a['unstable_modes']}  "
                  f"sub-gap modes (< sqrt2): {a['n_subgap']}  lowest freqs {np.round(a['slowest_10_freq'][:5], 3).tolist()}  "
                  f"wall weight sub-gap {a['subgap_wall_weight']:.2f} (x{a['subgap_wall_enrichment']:.0f} area) vs above-gap {a['abovegap_wall_weight']:.3f}")
        res[f"kappa_{kappa}"] = rows
    print("\nCoarsening from noise (64x64): number of domains (|phi| > 1 connected regions) over time")
    res["coarsening"] = {}
    for damping in [0.001, 0.1]:
        for kappa in [5.0, 0.0]:
            runs = [coarsening(kappa, damping, s) for s in SEEDS]
            res["coarsening"][f"damping{damping}_kappa{kappa}"] = runs
            for s, c in zip(SEEDS, runs):
                if c and c[-1][0] == "diverged":
                    print(f"  damping {damping} kappa {kappa} seed {s}: DIVERGED at step {c[-1][1]}"); continue
                pick = [c[i] for i in (0, 3, 9, 19, len(c) - 1) if i < len(c)]
                print(f"  damping {damping} kappa {kappa} seed {s}: " + "  ".join(f"t={t:.0f}:{n}" for t, n, _ in pick)
                      + f"   final rms velocity {c[-1][2]:.2e}")
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "g7a_phiworld_spectrum.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run()
