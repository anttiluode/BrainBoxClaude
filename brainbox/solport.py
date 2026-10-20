"""Port of BrainBoxSol's medium and writers (anttiluode/BrainBoxSol, branch
sol/local-material-credit-g0, commit 973bb63), generalized in ONE way:

each interface between adjacent blocks can have P parallel bridge edges whose
conductances sum to the same total. P = 1 is exactly Sol's medium; the port is
checked against his published numbers in gates/g4_sol_bridge_rescue.py.

The writers (write_one, write_protected) are Sol's update rules, unchanged.
"""
from dataclasses import dataclass, field
import numpy as np

# Bridge endpoint pairs (node in block k, node in block k+1). The first entry is
# Sol's original bridge: tip (3) of block k -> inlet (0) of block k+1.
BRIDGE_PAIRS = [(3, 0), (1, 1), (2, 2), (3, 1), (1, 0), (2, 0), (3, 2), (1, 2)]


@dataclass
class SolMedium:
    blocks: int = 4
    intra: float = 0.9
    bridge: float = 0.055          # TOTAL bridge conductance per interface
    leak: float = 0.035
    P: int = 1                     # parallel bridge edges per interface

    def __post_init__(self):
        edges, owner, iface = [], [], []
        for k in range(self.blocks):
            base = 4 * k
            for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
                edges.append((base + a, base + b)); owner.append(k); iface.append(-1)
            if k < self.blocks - 1:
                for a, b in BRIDGE_PAIRS[: self.P]:
                    edges.append((base + a, base + 4 + b)); owner.append(-1); iface.append(k)
        n = 4 * self.blocks
        B = np.zeros((n, len(edges)))
        for e, (a, b) in enumerate(edges):
            B[a, e] = 1; B[b, e] = -1
        self.incidence = B
        self.owner = np.asarray(owner)
        self.iface = np.asarray(iface)
        self.initial = np.where(self.owner == -1, self.bridge / self.P, self.intra).astype(float)
        self.ports = np.arange(self.blocks) * 4 + 3
        self.soma = 0

    def matrix(self, g):
        B = self.incidence
        return (B * g) @ B.T + self.leak * np.eye(B.shape[0])

    def read(self, g, block):
        A = self.matrix(g)
        src = np.zeros(A.shape[0]); src[self.ports[block]] = 1
        som = np.zeros(A.shape[0]); som[self.soma] = 1
        fwd = np.linalg.solve(A, src)
        adj = np.linalg.solve(A, som)
        d = -(self.incidence.T @ fwd) * (self.incidence.T @ adj)
        return float(fwd[self.soma]), d

    def values(self, g):
        return np.array([self.read(g, k)[0] for k in range(self.blocks)])


def teacher_medium(box, signs):
    t = box.initial.copy()
    for k, s in enumerate(signs):
        t[box.owner == k] *= 1 + 0.4 * s
    return t


def write_one(box, state, block, target, baseline, mode="local", rate=.15):
    """Sol's supervised adjoint step, unchanged."""
    response, sens = box.read(state, block)
    residual = (response - target[block]) / baseline[block]
    grad = sens / baseline[block]
    w = (box.owner == block) if mode == "local" else (box.owner >= 0)
    n2 = np.dot(grad[w], grad[w])
    state[w] -= rate * residual * grad[w] / max(n2, 1e-12)
    state[w] = np.clip(state[w], .1, 2.)


def write_protected(box, state, block, target, baseline, protected, rate=.15):
    """Sol's OGD/GPM-style protected writer, unchanged."""
    response, sens = box.read(state, block)
    residual = (response - target[block]) / baseline[block]
    w = box.owner >= 0
    cand = sens[w] / baseline[block]
    if protected:
        rows = np.stack([box.read(state, o)[1][w] / baseline[o] for o in protected])
        _, sv, Vt = np.linalg.svd(rows, full_matrices=False)
        act = sv > sv[0] * 1e-7
        if np.any(act):
            cand = cand - Vt[act].T @ (Vt[act] @ cand)
    m = float(cand @ cand)
    if m > 1e-12:
        state[w] -= rate * residual * cand / m
        state[w] = np.clip(state[w], .1, 2.)


def train(box, target, baseline, order, writer="local", steps=80, protected_steps=32):
    state = box.initial.copy()
    learned = []
    for k in order:
        if writer == "protected":
            for _ in range(protected_steps):
                write_protected(box, state, k, target, baseline, learned)
        else:
            for _ in range(steps):
                write_one(box, state, k, target, baseline, writer)
        learned.append(k)
    return state


# ------------------------------------------------------------- repairs after drift
def repair_replay(box, g, target, baseline, steps=160):
    """Sol's supervised repair: local writes with ALL old targets supplied."""
    g = g.copy()
    for s in range(steps):
        write_one(box, g, s % box.blocks, target, baseline, "local")
    return g


def repair_interface_total(box, g, g_ref):
    """Homeostat: each interface rescales its own bridges to its stored total.
    Stores one scalar per interface; uses no targets, no replay, no read-out."""
    g = g.copy()
    for k in range(box.blocks - 1):
        m = box.iface == k
        g[m] *= g_ref[m].sum() / g[m].sum()
    return g


def repair_node_scaling(box, g, g_ref, iters=200):
    """Control: classic per-node synaptic scaling toward each node's original total."""
    B = np.abs(box.incidence)
    target = B @ g_ref
    g = g.copy()
    for _ in range(iters):
        r = target / np.maximum(B @ g, 1e-12)
        a, b = (box.incidence > 0).argmax(0), (box.incidence < 0).argmax(0)
        g *= np.sqrt(r[a] * r[b])
    return g
