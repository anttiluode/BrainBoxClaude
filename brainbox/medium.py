"""A passive multi-tier medium: fast nodes grouped into sub-blocks, sub-blocks into
macro-blocks, coupled by progressively weaker "bridge" conductances.

Dynamics:  C dV/dt = -L V + I(t),  with C = 1 and L = graph Laplacian + leak.
Everything here is linear and solved exactly through the eigendecomposition of L,
so no integrator error enters any gate.
"""
from dataclasses import dataclass
import numpy as np
import scipy.linalg as sl


@dataclass
class Medium:
    L: np.ndarray          # N x N symmetric positive definite operator
    macro: np.ndarray      # macro-block label per node
    sub: np.ndarray        # global sub-block label per node
    lam: np.ndarray        # eigenvalues (ascending)
    phi: np.ndarray        # eigenvectors (columns)

    @property
    def N(self):
        return self.L.shape[0]


def build(M=4, S=4, n=8, g_in=1.0, g_sub=0.05, g_mac=0.0025, leak=1e-4,
          p_in=0.6, p_sub=0.15, p_mac=0.15, topology="all", seed=0):
    """Random hierarchical medium.

    topology="all":   every pair of macro-blocks may be bridged.
    topology="chain": macro-blocks bridged only to their neighbours (0-1-2-...).
    topology="flat":  no hierarchy; random graph with the same number of edges
                      and the same total conductance (control).
    """
    rng = np.random.default_rng(seed)
    N = M * S * n
    idx = np.arange(N)
    macro, sub = idx // (S * n), idx // n
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if sub[i] == sub[j]:
                g = g_in if rng.random() < p_in else 0.0
            elif macro[i] == macro[j]:
                g = g_sub if rng.random() < p_sub else 0.0
            else:
                if topology == "chain" and abs(macro[i] - macro[j]) != 1:
                    continue
                g = g_mac if rng.random() < p_mac else 0.0
            G[i, j] = G[j, i] = g
    if topology == "flat":
        w = G[np.triu_indices(N, 1)]
        w = w[w > 0]
        G = np.zeros((N, N))
        pairs = np.array(np.triu_indices(N, 1)).T
        pick = rng.choice(len(pairs), size=len(w), replace=False)
        for (i, j), g in zip(pairs[pick], rng.permutation(w)):
            G[i, j] = G[j, i] = g
    L = np.diag(G.sum(1)) - G + leak * np.eye(N)
    lam, phi = sl.eigh(L)
    return Medium(L, macro, sub, lam, phi)


def impulse(med: Medium, soma: int, ts: np.ndarray, sources=None):
    """h[i, t] = soma voltage at time t after a unit charge pulse at node i (t=0)."""
    if sources is None:
        sources = np.arange(med.N)
    w = med.phi[sources, :] * med.phi[soma, :][None, :]          # (S, K)
    return w @ np.exp(-np.outer(med.lam, ts))                     # (S, T)


def transfer(med: Medium, drive: int, omega: float):
    """Complex steady-state voltage at every node for a unit sinusoidal current
    I(t) = exp(i omega t) injected at `drive`:  V = (L + i omega)^-1 e_drive."""
    c = med.phi[drive, :] / (med.lam + 1j * omega)
    return med.phi @ c


def tiers(lam, min_ratio=2.0):
    """Split the eigenvalue spectrum into tiers at gaps whose ratio exceeds min_ratio."""
    lam = np.asarray(lam)
    cuts = [0]
    for k in range(1, len(lam)):
        if lam[k] / max(lam[k - 1], 1e-12) > min_ratio:
            cuts.append(k)
    cuts.append(len(lam))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def nnls_safe(A, y):
    """NNLS that never raises: generous iteration cap, then a clipped least-squares
    fallback for nearly-degenerate dictionaries (counted as a normal decode)."""
    from scipy.optimize import nnls
    try:
        return nnls(A, y, maxiter=50 * A.shape[1])[0]
    except RuntimeError:
        return np.clip(np.linalg.lstsq(A, y, rcond=None)[0], 0, None)
