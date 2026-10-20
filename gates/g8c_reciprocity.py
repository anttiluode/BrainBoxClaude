"""G8c - Is the return ping the adjoint? Reciprocity of PhiWorld's small-signal response.

Echo-based learning (G8b three-factor) uses the field radiated FROM an output as the sensitivity of
that output TO every point. That holds only if transmission is reciprocal: T(k -> j) = T(j -> k).
His rule c2(phi) * Lap(phi) is not the force of an energy, so its linearisation around a non-uniform
background is not symmetric. The variational form of the same rule is (brainbox/phimedium.py).

Measure: drive input k, read output j  vs  drive output j, read input k, all 3 x 3 pairs;
error = sum |T_kj - T'_jk| / sum |T_kj|. Media: uniform m (uniform background), random m
(m0 + 0.2 noise, clipped), and the m that G8b's three-factor writer produced (seed 0, if present).
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import g8_phiworld_wires as G
from brainbox.phimedium import P0

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def error(m, p):
    src_in, _, _ = G.ports(); w_in = src_in / src_in.sum((1, 2), keepdims=True)
    v = np.sqrt(p["a"] * (1 - m) / p["b"])
    phi, old = G.relax(v, v.copy(), m, p, 200.0)
    Af, Ae, T = G.fields(phi, old, m, p)
    Tb = np.einsum("jyx,kyx->kj", Ae, w_in)
    return float(np.abs(T - Tb).sum() / np.abs(T).sum())


def operator_checks():
    """Exact-gradient and symmetry checks of the variational clock on a random 8 x 9 field."""
    from brainbox.phimedium import clock_variational, lap5
    rng = np.random.default_rng(0); p = dict(P0)
    def E(phi):
        c2 = 1 / (1 + p["kappa"] * phi ** 2)
        gx = np.roll(phi, -1, -1) - phi; gy = np.roll(phi, -1, -2) - phi; gx[:, -1] = 0; gy[-1, :] = 0
        sx = 0.5 * (c2 + np.roll(c2, -1, -1)); sy = 0.5 * (c2 + np.roll(c2, -1, -2))
        return 0.5 * (sx * gx ** 2).sum() + 0.5 * (sy * gy ** 2).sum()
    phi = rng.standard_normal((8, 9)); h = 1e-6
    num = np.array([[-(E(phi + h * e) - E(phi - h * e)) / (2 * h) for e in np.eye(72)[i * 9:(i + 1) * 9].reshape(9, 8, 9)]
                    for i in range(8)])
    grad_err = float(np.abs(clock_variational(phi, p) - num).max())
    def jac(f):
        return np.stack([((f(phi + h * e.reshape(8, 9)) - f(phi - h * e.reshape(8, 9))) / (2 * h)).ravel() for e in np.eye(72)], 1)
    Jv = jac(lambda q: clock_variational(q, p)); Jo = jac(lambda q: (1 / (1 + p["kappa"] * q ** 2)) * lap5(q))
    out = dict(force_minus_dEdphi_max_err=grad_err,
               asymmetry_variational=float(np.abs(Jv - Jv.T).max() / np.abs(Jv).max()),
               asymmetry_his_form=float(np.abs(Jo - Jo.T).max() / np.abs(Jo).max()),
               kappa0_equals_lap5=float(np.abs(clock_variational(phi, dict(p, kappa=0.0)) - lap5(phi)).max()))
    print(out)
    return out


def run():
    checks = operator_checks()
    ms = {"uniform m": np.full((G.NY, G.NX), G.M0),
          "random m": np.clip(G.M0 + 0.2 * np.random.default_rng(0).standard_normal((G.NY, G.NX)), 0, 1)}
    for ai, lab in ((1, "trained m (variational)"), (0, "trained m (his form)")):
        f = os.path.join(OUT, f"g8b_m_arm{ai}_seed0.npy")
        if os.path.exists(f):
            ms[lab] = np.load(f)[0].astype(float)
    rows = []
    for form, kappa in (("phiworld", 5.0), ("variational", 5.0), ("phiworld", 0.0)):
        p = dict(P0, form=form, kappa=kappa)
        for name, m in ms.items():
            e = error(m, p)
            rows.append(dict(form=form, kappa=kappa, medium=name, reciprocity_error=e))
            print(f"{form:11s} kappa {kappa}: {name:24s} reciprocity error {e:.3f}")
    json.dump(dict(operator_checks=checks, reciprocity=rows), open(os.path.join(OUT, "g8c_reciprocity.json"), "w"), indent=1)


if __name__ == "__main__":
    run()
