"""PhiWorld medium with a slow material field m (G8).

Fast field (his phiworld2 equations, leapfrog update copied from his code):
    phi_tt = c2(phi) * D_sigma(phi) - V'_m(phi) - gamma * Lap^2(phi) + F(x, t)
    c2(phi) = c0 / (1 + kappa * phi^2)           (mass slows the clock)
    V'_m(phi) = -a * (1 - mu_gap * m) * phi + b * phi^3
    new = phi + (1 - damping*dt) * (phi - old) + dt^2 * accel        (his update)

Two ways m can couple (tested against each other in G8):
  * 'gap'  : m flattens the Mexican hat locally (a -> a(1 - mu_gap m)). Where m = 1 the
             local mass gap closes, so a stripe of m is a gap-free channel inside gapped domains.
  * 'cond' : m raises conductance in the divergence-form operator D_sigma = div(sigma grad phi),
             sigma = 1 + m (alpha t t^T + beta n n^T), t = wall tangent (Grok's idea, corrected).

All operators act on the last two axes, so a batch of independent runs is one array (B, Ny, Nx).
Boundaries: 'neumann' (zero flux, used in G8 so ports have a left and a right) or 'periodic'.
"""
import numpy as np

P0 = dict(c0=1.0, kappa=5.0, a=1.0, b=0.2, gamma=0.02, dt=0.08, damping=0.001)


# ----------------------------------------------------------------- operators
def lap5(phi, bc="neumann"):
    """Standard 5-point Laplacian (grid spacing 1)."""
    if bc == "periodic":
        return (np.roll(phi, 1, -1) + np.roll(phi, -1, -1) + np.roll(phi, 1, -2) + np.roll(phi, -1, -2) - 4 * phi)
    p = np.pad(phi, [(0, 0)] * (phi.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    return p[..., 1:-1, 2:] + p[..., 1:-1, :-2] + p[..., 2:, 1:-1] + p[..., :-2, 1:-1] - 4 * phi


def face_fluxes(phi, sx, sy, bc="neumann"):
    """Staggered (face-centred) fluxes. sx: conductance on x-faces between (i,j) and (i,j+1),
    sy: on y-faces between (i,j) and (i+1,j). Shapes = phi's shape (last face unused for neumann)."""
    gx = np.roll(phi, -1, -1) - phi            # forward difference -> lives on the face (j + 1/2)
    gy = np.roll(phi, -1, -2) - phi
    fx, fy = sx * gx, sy * gy
    if bc == "neumann":                        # no flux through the outer boundary
        fx = fx.copy(); fy = fy.copy()
        fx[..., :, -1] = 0.0
        fy[..., -1, :] = 0.0
    return fx, fy


def div_sigma_grad(phi, sx, sy, bc="neumann"):
    """D_sigma phi = div(sigma grad phi) with face conductances. sigma = 1 -> exactly lap5."""
    fx, fy = face_fluxes(phi, sx, sy, bc)
    return fx - np.roll(fx, 1, -1) + fy - np.roll(fy, 1, -2) if bc == "periodic" else \
        _div_neumann(fx, fy)


def _div_neumann(fx, fy):
    dx = fx.copy(); dx[..., :, 1:] -= fx[..., :, :-1]
    dy = fy.copy(); dy[..., 1:, :] -= fy[..., :-1, :]
    return dx + dy


def tangent_conductance(phi, m, alpha=8.0, beta=0.3, bc="neumann"):
    """Diagonal of sigma = 1 + m (alpha t t^T + beta n n^T) at the faces, n = grad phi / |grad phi|,
    t = n rotated 90 degrees. Strong conduction ALONG walls (t), weak across (n).
    (Grok's version had alpha on n n^T, i.e. strong conduction across the wall.)
    Off-diagonal sigma_xy is dropped (documented approximation; exact on axis-aligned walls)."""
    if bc == "periodic":
        gx = 0.5 * (np.roll(phi, -1, -1) - np.roll(phi, 1, -1)); gy = 0.5 * (np.roll(phi, -1, -2) - np.roll(phi, 1, -2))
    else:
        gy, gx = np.gradient(phi, axis=(-2, -1))
    g = np.sqrt(gx ** 2 + gy ** 2) + 1e-9
    nx2, ny2 = (gx / g) ** 2, (gy / g) ** 2
    tx2, ty2 = ny2, nx2
    sxx = 1 + m * (alpha * tx2 + beta * nx2)
    syy = 1 + m * (alpha * ty2 + beta * ny2)
    sx = 0.5 * (sxx + np.roll(sxx, -1, -1)); sy = 0.5 * (syy + np.roll(syy, -1, -2))
    return sx, sy


def grok_brain_laplacian(phi, m, alpha=8.0, beta=0.3, sigma0=1.0):
    """Grok's code, verbatim (periodic). Kept only to test it."""
    gx = np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)
    gy = np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)
    g2 = gx * gx + gy * gy + 1e-12
    nx, ny = gx / np.sqrt(g2), gy / np.sqrt(g2); tx, ty = -ny, nx
    s_xx = sigma0 + m * (alpha * nx * nx + beta * tx * tx)
    s_yy = sigma0 + m * (alpha * ny * ny + beta * ty * ty)
    s_xy = m * (alpha * nx * ny + beta * tx * ty)
    fx = s_xx * gx + s_xy * gy; fy = s_xy * gx + s_yy * gy
    return (np.roll(fx, 1, axis=1) - np.roll(fx, -1, axis=1) + np.roll(fy, 1, axis=0) - np.roll(fy, -1, axis=0)) / 4.0


# ----------------------------------------------------------------- dynamics
def clock_variational(phi, p, bc="neumann"):
    """The same 'mass slows the clock' rule, but as the force of an energy:
        E = sum_faces 1/2 c2_face (grad phi)^2,  c2_face = mean of c2(phi) at the face's two cells
        force = -dE/dphi = div(c2_face grad phi) - 1/4 c2'(phi) * sum_adjacent_faces (grad phi)^2
    Continuum: div(c2 grad phi) - 1/2 c2' |grad phi|^2. Its linearisation is a Hessian (symmetric),
    so small signals are reciprocal: the echo from an output IS the adjoint field. His original
    c2(phi) * Lap(phi) is not the force of any energy and is not reciprocal (G8b)."""
    c2 = p["c0"] / (1 + p["kappa"] * phi ** 2)
    dc2 = -2 * p["kappa"] * p["c0"] * phi / (1 + p["kappa"] * phi ** 2) ** 2
    sx = 0.5 * (c2 + np.roll(c2, -1, -1)); sy = 0.5 * (c2 + np.roll(c2, -1, -2))
    gx = np.roll(phi, -1, -1) - phi; gy = np.roll(phi, -1, -2) - phi
    if bc == "neumann":
        gx = gx.copy(); gy = gy.copy(); gx[..., :, -1] = 0.0; gy[..., -1, :] = 0.0
    fx, fy = sx * gx, sy * gy
    div = fx - np.roll(fx, 1, -1) + fy - np.roll(fy, 1, -2) if bc == "periodic" else _div_neumann(fx, fy)
    gx2, gy2 = gx ** 2, gy ** 2
    if bc == "periodic":
        S = gx2 + np.roll(gx2, 1, -1) + gy2 + np.roll(gy2, 1, -2)
    else:
        S = gx2.copy(); S[..., :, 1:] += gx2[..., :, :-1]; S += gy2; S[..., 1:, :] += gy2[..., :-1, :]
    return div - 0.25 * dc2 * S


def accel(phi, m, p, coupling="gap", bc="neumann", force=None):
    c2 = p["c0"] / (1 + p["kappa"] * phi ** 2)
    if p.get("form", "phiworld") == "variational":
        lp = lap5(phi, bc)
        a_eff = p["a"] * (1 - p.get("mu_gap", 1.0) * m) if (coupling == "gap" and m is not None) else p["a"]
        acc = clock_variational(phi, p, bc) - (-a_eff * phi + p["b"] * phi ** 3) - p["gamma"] * lap5(lp, bc)
        return acc if force is None else acc + force
    if coupling == "cond" and m is not None:
        sx, sy = tangent_conductance(phi, m, bc=bc)
        wave = div_sigma_grad(phi, sx, sy, bc)
    else:
        wave = lap5(phi, bc)
    lp = lap5(phi, bc)
    a_eff = p["a"] * (1 - p.get("mu_gap", 1.0) * m) if (coupling == "gap" and m is not None) else p["a"]
    acc = c2 * wave - (-a_eff * phi + p["b"] * phi ** 3) - p["gamma"] * lap5(lp, bc)
    if force is not None:
        acc = acc + force
    return acc


def step(phi, old, m, p, coupling="gap", bc="neumann", force=None):
    new = phi + (1 - p["damping"] * p["dt"]) * (phi - old) + p["dt"] ** 2 * accel(phi, m, p, coupling, bc, force)
    return new, phi


def evolve(phi, old, m, p, t, coupling="gap", bc="neumann"):
    for _ in range(int(round(t / p["dt"]))):
        phi, old = step(phi, old, m, p, coupling, bc)
    return phi, old


def noise_init(rng, shape, amp=0.05):
    phi = (rng.random(shape) - 0.5) * 2 * amp
    return phi, phi.copy()


def blob(shape, cy, cx, r=1.5):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * r * r))


def drive_lockin(phi0, old0, m, p, sources, omega, amp, t_settle, t_meas, coupling="gap", bc="neumann",
                 probes=None):
    """Batch of runs from the same starting state. sources: (B, Ny, Nx) spatial drive profiles.
    Each run is driven by amp * src * sin(omega t). Returns the complex lock-in amplitude of phi at
    omega over a whole number of periods, for every pixel (B, Ny, Nx), with the starting state
    subtracted (a static background times sin/cos would otherwise leak in; G8 ledger 9).
    Meant for a cold (static) background; in the hot state the background swamps the signal."""
    B = sources.shape[0]
    if phi0.ndim == 3 and phi0.shape[0] == B:          # one starting state per run
        phi, old = phi0.copy(), old0.copy()
    else:
        phi = np.repeat(phi0[None], B, 0); old = np.repeat(old0[None], B, 0)
    dt = p["dt"]
    period = 2 * np.pi / omega
    n_set = int(round(t_settle / dt))
    n_meas = int(round(max(1, round(t_meas / period)) * period / dt))   # whole number of periods
    ref = phi.copy()                                                     # static background removed
    acc_c = np.zeros_like(phi); acc_s = np.zeros_like(phi)
    for k in range(n_set + n_meas):
        t = k * dt
        f = amp * sources * np.sin(omega * t)
        phi, old = step(phi, old, m, p, coupling, bc, f)
        if k >= n_set:
            d = phi - ref
            acc_c += d * np.cos(omega * t); acc_s += d * np.sin(omega * t)
    return 2 * (acc_s + 1j * acc_c) / n_meas, phi
