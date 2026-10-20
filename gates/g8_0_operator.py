"""G8.0 - Grok's brain_laplacian vs a corrected divergence-form operator (unit tests, no dynamics).

At m = 0 any div(sigma grad phi) operator must reduce to the plain Laplacian his PhiWorld uses.
Checks: sign/shape on a smooth Gaussian, the finest grid mode (checkerboard), conservation (sum = 0),
self-adjointness, and which direction the tensor conducts on a straight wall.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox.phimedium import lap5, div_sigma_grad, tangent_conductance, grok_brain_laplacian


def grok_sigma_on_wall(wall):
    """Grok's tensor diagonal on the wall (alpha on n n^T)."""
    gy, gx = np.gradient(wall)
    g = np.sqrt(gx ** 2 + gy ** 2) + 1e-12
    nx, ny = gx / g, gy / g
    return 1 + 8.0 * nx ** 2 + 0.3 * ny ** 2, 1 + 8.0 * ny ** 2 + 0.3 * nx ** 2   # sigma_xx, sigma_yy


def run():
    n = 48
    y, x = np.mgrid[:n, :n]
    gauss = np.exp(-((x - 24) ** 2 + (y - 24) ** 2) / (2 * 4.0 ** 2))
    cb = ((x + y) % 2) * 2 - 1.0
    one = np.ones((n, n)); zero = np.zeros((n, n))
    rng = np.random.default_rng(0)
    s = rng.random((n, n)) + 0.5
    u, v = rng.standard_normal((2, n, n))
    true_g, grok_g, ours_g = lap5(gauss, "periodic"), grok_brain_laplacian(gauss, zero), div_sigma_grad(gauss, one, one, "periodic")
    wall = np.sqrt(5) * np.tanh((x - 24) / 2.0)                    # vertical wall: tangent = y
    sx, sy = tangent_conductance(wall, one, bc="periodic")
    gxx, gyy = grok_sigma_on_wall(wall)
    res = {
        "gaussian_centre": dict(true=float(true_g[24, 24]), grok=float(grok_g[24, 24]), ours=float(ours_g[24, 24])),
        "gaussian_corr_with_true": dict(grok=float(np.corrcoef(grok_g.ravel(), true_g.ravel())[0, 1]),
                                        ours=float(np.corrcoef(ours_g.ravel(), true_g.ravel())[0, 1])),
        "checkerboard": dict(true=float(lap5(cb, "periodic")[10, 10]), grok=float(grok_brain_laplacian(cb, zero)[10, 10]),
                             ours=float(div_sigma_grad(cb, one, one, "periodic")[10, 10])),
        "ours_sigma1_max_abs_diff_from_lap5": float(np.abs(div_sigma_grad(u, one, one, "periodic") - lap5(u, "periodic")).max()),
        "ours_sum_with_random_sigma": float(div_sigma_grad(u, s, s, "periodic").sum()),
        "ours_self_adjoint_gap": float((u * div_sigma_grad(v, s, s, "periodic")).sum() - (v * div_sigma_grad(u, s, s, "periodic")).sum()),
        "on_vertical_wall_m1": dict(ours_across=float(sx[24, 24]), ours_along=float(sy[24, 24]),
                                    grok_across=float(gxx[24, 24]), grok_along=float(gyy[24, 24])),
    }
    for k, v_ in res.items():
        print(f"{k}: {v_}")
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "g8_0_operator.json"), "w"), indent=1)


if __name__ == "__main__":
    run()
