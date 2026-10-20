"""G8b runner - every writer x medium, 3 seeds, crossing mapping pi = (1, 2, 0). See g8_phiworld_wires.py.

Pre-stated (written after the G8b prototypes in the ledger, before this run):
  Q1 three-factor in a reciprocal medium (variational kappa = 5, and kappa = 0) learns the crossing
     mapping: accuracy 1.0 on >= 2/3 seeds, well above chance 1/3.
  Q2 three-factor in his original form (c2(phi) Lap phi, not reciprocal) does worse than in the
     variational form of the same rule.
  Q3 Hebbian coincidence, forward-only, grok and no-learning stay near chance on the crossing mapping.
  Q4 the trained m alone carries the mapping: re-born phi (new noise, hot, cooled, m kept) keeps the
     accuracy; spatially shuffled m loses it; training on the wrong mapping gives ~0.
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import g8_phiworld_wires as G
from brainbox.phimedium import P0

ARMS = [  # name, rule, form, kappa, uniform start, train on
    ("three-factor | his form k=5", "three-factor", "phiworld", 5.0, False, G.PI_CYCLIC),
    ("three-factor | variational k=5", "three-factor", "variational", 5.0, False, G.PI_CYCLIC),
    ("three-factor | k=0", "three-factor", "phiworld", 0.0, False, G.PI_CYCLIC),
    ("three-factor | variational k=5, uniform start", "three-factor", "variational", 5.0, True, G.PI_CYCLIC),
    ("three-factor | variational k=5, WRONG mapping", "three-factor", "variational", 5.0, False, G.PI_WRONG[G.PI_CYCLIC]),
    ("coincidence (Hebbian) | variational k=5", "coincidence", "variational", 5.0, False, G.PI_CYCLIC),
    ("coincidence (Hebbian) | his form k=5", "coincidence", "phiworld", 5.0, False, G.PI_CYCLIC),
    ("forward only | variational k=5", "forward", "variational", 5.0, False, G.PI_CYCLIC),
    ("grok |grad phi| | variational k=5", "grok", "variational", 5.0, False, G.PI_CYCLIC),
    ("no learning | variational k=5", "none", "variational", 5.0, False, G.PI_CYCLIC),
]
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def job(args):
    ai, seed = args
    name, rule, form, kappa, uniform, pi_train = ARMS[ai]
    p = dict(P0, form=form, kappa=kappa)
    t = time.time()
    m, phi, old, log = G.train(rule, pi_train, seed, p, uniform=uniform, pi_log=G.PI_CYCLIC)
    T = G.fields(phi, old, m, p, echo=False)[2]
    res = dict(arm=name, seed=seed, final=G.score(T, G.PI_CYCLIC), log=log, T=np.abs(T / G.AMP).tolist())
    P = np.abs(T)
    res["final"]["contrast"] = float(np.mean([P[k, G.PI_CYCLIC[k]] / np.mean([P[k, j] for j in range(G.K) if j != G.PI_CYCLIC[k]])
                                              for k in range(G.K)]))
    if rule == "three-factor" and not uniform and pi_train == G.PI_CYCLIC:
        rng = np.random.default_rng(1000 + seed)
        ph2, o2 = G.born(rng, m, p)                                     # phi re-born, m kept
        res["reborn"] = G.score(G.fields(ph2, o2, m, p, echo=False)[2], G.PI_CYCLIC)
        ms = rng.permutation(m.ravel()).reshape(m.shape)                 # same material, scrambled
        ph3, o3 = G.relax(phi, old, ms, p, 200.0)
        res["shuffled"] = G.score(G.fields(ph3, o3, ms, p, echo=False)[2], G.PI_CYCLIC)
        np.save(os.path.join(OUT, f"g8b_m_arm{ai}_seed{seed}.npy"), np.stack([m, phi]).astype(np.float32))
    res["seconds"] = round(time.time() - t)
    print(f"[{name}] seed {seed}: {res['final']}  " + (f"reborn {res['reborn']}  shuffled {res['shuffled']}" if "reborn" in res else "")
          + f"  ({res['seconds']}s)", flush=True)
    return res


if __name__ == "__main__":
    jobs = [(ai, s) for s in G.SEEDS for ai in range(len(ARMS))]
    with Pool(2) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    json.dump(dict(config=dict(NY=G.NY, NX=G.NX, K=G.K, M0=G.M0, OMEGA=G.OMEGA, AMP=G.AMP, DAMP=G.DAMP,
                               EPOCHS=G.EPOCHS, ETA=G.ETA, SEEDS=G.SEEDS, PI=G.PI_CYCLIC), rows=rows),
              open(os.path.join(OUT, "g8b_learning.json"), "w"), indent=1)
    print("\nsummary (mean over seeds): accuracy | log p(target) | T_target | contrast")
    for ai, a in enumerate(ARMS):
        r = [x for x in rows if x["arm"] == a[0]]
        f = lambda k: np.mean([x["final"][k] for x in r])
        accs = [x["final"]["acc"] for x in r]
        line = f"{a[0]:48s} acc {f('acc'):.2f} {np.round(accs, 2).tolist()}  logp {f('logp'):+.2f}  T {f('T_target'):.3f}  contrast {f('contrast'):.1f}"
        if "reborn" in r[0]:
            line += f"  | reborn acc {np.mean([x['reborn']['acc'] for x in r]):.2f}  shuffled acc {np.mean([x['shuffled']['acc'] for x in r]):.2f}"
        print(line)
