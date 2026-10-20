"""G8b - Can PhiWorld wire itself? A slow material field m written by activity, tested as routing.

Medium: his phiworld2 equations (brainbox/phimedium.py) + a slow field m that flattens the local
Mexican hat, a -> a(1 - m). In a domain the small-signal gap is sqrt(2 a (1 - m)): low m insulates at
the drive frequency, high m conducts (G8a: an m = 1 stripe carries the signal, a bare domain carries
nothing). Absorbing (sponge) edges, damping 0.005.

Task: K = 3 inputs (left), 3 outputs (right), mapping pi = (1, 2, 0) (every wire must cross another in
2-D). It is taught ONLY by driving the medium: forward runs (input k driven at omega) and echo runs
(output j driven at omega - the 'return ping'). Read-out: drive input k alone, lock-in amplitude at
every output -> T[k, j]. Accuracy = fraction of inputs whose strongest output is pi(k) (chance 1/3).

The total amount of m is conserved (sum m = m0 * area, m0 = 0.7): writers can only move material.
Every writer moves it with the same step (eta, scaled by the 95th percentile of its own update).

Writers:
  three-factor : dm(x) = sum_kj w_kj Re(conj(T_kj) A_fwd,k(x) A_echo,j(x))
                 local phase-sensitive coincidence of forward and echo fields x one outcome scalar per
                 pair (w = 2/|T_k,pi(k)|^2 for the target, -1/sum_j |T_kj|^2 for every output).
                 If the medium is reciprocal (echo = adjoint field) this is the gradient of
                 sum_k [2 log|T_k,pi(k)|^2 - log sum_j |T_kj|^2]. Prior art: adjoint / in-situ training of
                 wave media (Hughes et al. 2018, 2019).
  coincidence  : dm ~ |A_fwd,k| |A_echo,pi(k)|, each pair normalised to the same total (Hebbian, no outcome)
  forward      : dm ~ |A_fwd,k|^2 (activity only, no echo)
  grok         : dm ~ |grad phi| (all walls; no drive, no echo)
  none         : m stays uniform
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox.phimedium import P0, blob, drive_lockin, evolve, noise_init

NY, NX, K = 32, 48, 3
PORT_Y = [6.0, 15.5, 25.0]
X_IN, X_OUT = 8, 39
M0 = 0.7
OMEGA, AMP = 1.0, 0.02          # above the m = 0.7 gap (0.77), below the m = 0 gap (1.41)
T_SET, T_MEAS = 150.0, 60.0
DAMP = 0.005
SPONGE = 0.4
EPOCHS, ETA = int(os.environ.get("G8_EPOCHS", 60)), float(os.environ.get("G8_ETA", 0.01))
SIGN = float(os.environ.get("G8_SIGN", +1))   # +1 = the derived ascent direction (ledger 9)
PRUNE = tuple(int(v) for v in os.environ.get("G8_PRUNE", "1000,1001").split(","))   # default: never prune (total conserved)
WIRE_CELLS = 4 * (X_OUT - X_IN)
SEEDS = [0, 1, 2]
PI_CYCLIC, PI_ID = (1, 2, 0), (0, 1, 2)
PI_WRONG = {PI_CYCLIC: (2, 0, 1), PI_ID: (2, 1, 0)}

_y, _x = np.mgrid[:NY, :NX]
_edge = np.minimum.reduce([_x, NX - 1 - _x, _y, NY - 1 - _y]).astype(float)
DAMP_MAP = DAMP + SPONGE * np.clip((4 - _edge) / 4, 0, 1) ** 2


def ports():
    src_in = np.stack([blob((NY, NX), y, X_IN) for y in PORT_Y])
    src_out = np.stack([blob((NY, NX), y, X_OUT) for y in PORT_Y])
    return src_in, src_out, src_out / src_out.sum((1, 2), keepdims=True)


def project(m, budget):
    """Closest m in {0 <= m <= 1, sum m = budget}: clip(m - tau)."""
    lo, hi = m.min() - 1.0, m.max()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.clip(m - mid, 0, 1).sum() > budget: lo = mid
        else: hi = mid
    return np.clip(m - hi, 0, 1)


def relax(phi, old, m, p, t=60.0):
    return evolve(phi, old, m, dict(p, damping=0.3), t)


def born(rng, m, p, uniform=False):
    """PhiWorld birth: his noise init, hot (damping 0.0025), then cooled. Or one uniform domain."""
    if uniform:
        v = np.sqrt(p["a"] * (1 - m) / p["b"]); phi, old = v.copy(), v.copy()
    else:
        phi, old = noise_init(rng, (NY, NX))
        phi, old = evolve(phi, old, m, dict(p, damping=0.0025), 300.0)
    return relax(phi, old, m, p, 200.0)


def fields(phi, old, m, p, echo=True):
    src_in, src_out, w_out = ports()
    srcs = np.concatenate([src_in, src_out]) if echo else src_in
    A, _ = drive_lockin(phi, old, m, dict(p, damping=DAMP_MAP), srcs, OMEGA, AMP, T_SET, T_MEAS)
    T = np.einsum("kyx,jyx->kj", A[:K], w_out)
    return A[:K], (A[K:] if echo else None), T


def score(T, pi):
    P = np.abs(T)
    acc = float(np.mean([np.argmax(P[k]) == pi[k] for k in range(K)]))
    logit = float(np.mean([np.log(P[k, pi[k]] ** 2 / (P[k] ** 2).sum()) for k in range(K)]))
    return dict(acc=acc, logp=logit, T_target=float(np.mean([P[k, pi[k]] for k in range(K)])) / AMP)


def writer(rule, phi, old, m, p, pi):
    if rule == "grok":
        gy, gx = np.gradient(phi)
        return np.sqrt(gx ** 2 + gy ** 2)
    Af, Ae, T = fields(phi, old, m, p, echo=(rule != "forward"))
    if rule == "forward":
        return (np.abs(Af) ** 2).sum(0)
    if rule == "coincidence":       # hebbian, but every pair writes the same total (no pair can win)
        return sum(c / (c.sum() + 1e-30) for c in (np.abs(Af[k]) * np.abs(Ae[pi[k]]) for k in range(K)))
    P = np.abs(T) ** 2
    g = np.zeros((NY, NX))
    for k in range(K):
        for j in range(K):
            gj = np.real(np.conj(T[k, j]) * Af[k] * Ae[j])
            g += (2 * gj / P[k, pi[k]] if j == pi[k] else 0.0) - gj / P[k].sum()
    return SIGN * g


def train(rule, pi_train, seed, p, uniform=False, epochs=EPOCHS, pi_log=None):
    rng = np.random.default_rng(seed)
    m = np.full((NY, NX), M0)
    phi, old = born(rng, m, p, uniform)
    b0, bf = M0 * NY * NX, K * WIRE_CELLS
    log = []
    for e in range(epochs):
        budget = bf + (b0 - bf) * np.clip(1 - (e + 1 - PRUNE[0]) / (PRUNE[1] - PRUNE[0]), 0, 1)
        if rule != "none":
            d = writer(rule, phi, old, m, p, pi_train)
            m = m + ETA * d / (np.percentile(np.abs(d), 95) + 1e-30)
        m = project(m, budget)
        phi, old = relax(phi, old, m, p)
        if pi_log is not None and ((e + 1) % max(1, epochs // 5) == 0):
            log.append((e + 1, score(fields(phi, old, m, p, echo=False)[2], pi_log)))
    return m, phi, old, log


if __name__ == "__main__" and os.environ.get("G8_PROTO"):
    p = dict(P0, form=os.environ.get("G8_FORM", "phiworld"), kappa=float(os.environ.get("G8_KAPPA", 5)))
    rule = os.environ.get("G8_RULE", "three-factor")
    for pi in [PI_CYCLIC] if os.environ.get("G8_ONLYCYC") else [PI_ID, PI_CYCLIC]:
        t = time.time()
        m, phi, old, log = train(rule, pi, 0, p, pi_log=pi)
        print(rule, pi, f"{time.time() - t:.0f}s")
        for e, s in log: print("  epoch", e, {k: round(v, 3) for k, v in s.items()})
        T = fields(phi, old, m, p, echo=False)[2]; print(np.round(np.abs(T) / AMP, 4))
        np.save(f"/tmp/claude-0/-home-claude/cdc1a9f1-fb9f-5335-9df1-4f5a68017549/scratchpad/m_{rule}{pi[0]}_{p['form']}{p['kappa']}.npy", np.stack([m, phi]))
