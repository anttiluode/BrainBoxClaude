"""G10b - canalization: can a rich-get-richer material flow turn the faint learned routes into channels?

Gemini's proposal (Oct 20 evening): the three-factor writer stops at a faint pattern (target transmission 0.18,
vs ~1.07 for a hand-drawn m = 1 stripe, G8a) because it only needs the target to win the argmax. Rivers and
self-focusing beams sharpen because material flows toward where the flux is. Add that flow:

    dm = [three-factor writer, as G8b]  -  div J,     J = chi * m (1 - m) * grad E  -  D grad m
    E(x) = forward-wave energy, sum_k |A_fwd,k(x)|^2, each input normalised, clipped at its 95th percentile

The flow term is conservative (div form, zero flux at the edges): it moves material up the energy gradient
and needs no global projection. Same box, task and writer as G8b (variational kappa = 5), but every field is
read after 400 t of settling (G9a: 150 t was not settled), so chi = 0 is also the first SETTLED-gradient G8b.

Pre-stated:
  C1 chi > 0 raises target transmission above chi = 0 without lowering accuracy.
  C2 (Gemini's number) target transmission >= 0.8 with full accuracy. My expectation: not reached, because the
     crossing mapping forces channels to cross and crossings leak (G8a stripes leak nothing only when alone).
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import g8_phiworld_wires as G
from brainbox.phimedium import P0

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
G.T_SET = 400.0
D = 0.05
ARMS = [("chi = 0 (settled G8b)", 0.0), ("chi = 0.05", 0.05), ("chi = 0.2", 0.2)]


def flow_step(m, E, chi):
    """m <- m - div J on the face grid, J = chi m(1-m) grad E - D grad m, zero flux at the boundary."""
    mf_x = 0.5 * (m[:, 1:] + m[:, :-1]); mf_y = 0.5 * (m[1:, :] + m[:-1, :])
    Jx = chi * mf_x * (1 - mf_x) * (E[:, 1:] - E[:, :-1]) - D * (m[:, 1:] - m[:, :-1])
    Jy = chi * mf_y * (1 - mf_y) * (E[1:, :] - E[:-1, :]) - D * (m[1:, :] - m[:-1, :])
    div = np.zeros_like(m)
    div[:, :-1] += Jx; div[:, 1:] -= Jx
    div[:-1, :] += Jy; div[1:, :] -= Jy
    return np.clip(m - div, 0, 1)


def train(seed, chi, p, epochs=G.EPOCHS):
    rng = np.random.default_rng(seed)
    m = np.full((G.NY, G.NX), G.M0)
    phi, old = G.born(rng, m, p)
    budget = G.M0 * G.NY * G.NX
    for e in range(epochs):
        Af, Ae, T = G.fields(phi, old, m, p)
        P = np.abs(T) ** 2
        g = np.zeros_like(m)
        for k in range(G.K):
            for j in range(G.K):
                gj = np.real(np.conj(T[k, j]) * Af[k] * Ae[j])
                g += (2 * gj / P[k, G.PI_CYCLIC[k]] if j == G.PI_CYCLIC[k] else 0.0) - gj / P[k].sum()
        m = G.project(m + G.ETA * g / (np.percentile(np.abs(g), 95) + 1e-30), budget)
        if chi > 0:
            E = sum(np.minimum(a / np.percentile(a, 95), 1.0) for a in (np.abs(Af) ** 2))
            m = flow_step(m, E, chi)
            m = G.project(m, budget)            # the flow conserves; this only undoes clipping
        phi, old = G.relax(phi, old, m, p)
    return m, phi, old


def job(args):
    ai, seed = args
    name, chi = ARMS[ai]
    p = dict(P0, form="variational", kappa=5.0)
    t = time.time()
    m, phi, old = train(seed, chi, p)
    T = G.fields(phi, old, m, p, echo=False)[2]
    s = G.score(T, G.PI_CYCLIC)
    rng = np.random.default_rng(1000 + seed)
    ph2, o2 = G.born(rng, m, p)
    s["reborn_acc"] = G.score(G.fields(ph2, o2, m, p, echo=False)[2], G.PI_CYCLIC)["acc"]
    np.save(os.path.join(OUT, f"g10b_m_arm{ai}_seed{seed}.npy"), np.stack([m, phi]).astype(np.float32))
    print(f"[{name}] seed {seed}: acc {s['acc']:.2f}  T_target {s['T_target']:.3f}  logp {s['logp']:+.2f}  reborn {s['reborn_acc']:.2f}"
          f"  T {np.round(np.abs(T) / G.AMP, 3).tolist()}  ({time.time() - t:.0f}s)", flush=True)
    return dict(arm=name, chi=chi, seed=seed, final=s, T=(np.abs(T) / G.AMP).tolist())


if __name__ == "__main__":
    jobs = [(ai, s) for s in G.SEEDS for ai in range(len(ARMS))]
    with Pool(2) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    json.dump(dict(config=dict(D=D, T_SET=G.T_SET, EPOCHS=G.EPOCHS, ETA=G.ETA), rows=rows),
              open(os.path.join(OUT, "g10b_canalization.json"), "w"), indent=1)
    for name, chi in ARMS:
        r = [x for x in rows if x["arm"] == name]
        print(f"{name:24s} acc {np.mean([x['final']['acc'] for x in r]):.2f} {[x['final']['acc'] for x in r]}  "
              f"T {np.mean([x['final']['T_target'] for x in r]):.3f}  reborn {np.mean([x['final']['reborn_acc'] for x in r]):.2f}")
