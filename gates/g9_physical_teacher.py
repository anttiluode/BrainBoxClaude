"""G9 - make the teacher physical.

G8b's writer was computed outside the medium: forward and echo were separate runs from a copied state,
and a per-pair outcome number was broadcast to every pixel. Here everything happens in ONE medium, in
time, with one shared clock:

  for each input k, three phases in a row (no copies, the field just keeps running):
    free    : input k driven, AMP sin(w t).  Each output j measures its own received phasor T_kj
              (a lock-in at the output, using the clock).
    nudged  : input k driven AND every output re-emits at w with source phasor
                  s_j = beta * AMP * sigma_j * conj(T_kj) / |T_kj|
              sigma_j = +1 if j is k's target, -1 otherwise (the label lives at the output).
    echo    : the outputs re-emit s_j alone, input silent.
  the material at x senses only its own activity delta = phi - <phi> (deviation from its local mean)
  and the clock at 2w:   Q(x) = 2w lock-in of delta(x, t)^2.
  write:   dm(x) ~ Re( 2i [Q_nudged - Q_free - Q_echo] )   (+ the conserved-total projection)

Why the 2w part: for delta = Im(A e^{iwt}) the 2w lock-in of delta^2 is -(i/2) A^2, so
2i(Q_nudged - Q_free - Q_echo) = 2 A_fwd * A_echo  - the product of the two fields with NO complex
conjugate, which is what the gradient needs (G8b: Re(conj(T) A_fwd A_echo)). The plain intensity
(the DC part of delta^2) gives A_fwd * conj(A_echo) instead - the wrong product. That is the 'no clock'
control. A 2-phase variant (free, nudged; no echo-only phase) keeps a beta^2 echo self-term.

Objective implied by the echoes (echo_weights): maximise sum_k ( log|T_k,target| - mean_{j != target} log|T_kj| ),
with the echo gain capped (CAP). v1 used sigma = +-1 with unit-loudness echoes, i.e. the AMPLITUDE objective
sum_k (|T_k,target| - sum_other |T_kj|): it failed R1 because one pair won (ledger 14, results/g9b_v1_*.log).
Pre-stated for G9b v2 (written after v1 failed, before the v2 run; same statements as v1):
  R1 the 3-phase physical loop learns the crossing mapping in the variational medium on >= 2/3 seeds.
  R2 the 'no clock' (plain intensity) writer stays near chance (G9a: its correlation with the gradient ~0).
  R3 wrong labels -> ~0; reborn phi keeps the routing, shuffled m loses it.
  R4 (open, no prediction) 2-phase and his non-reciprocal form.
Still outside the medium, stated plainly: the clock (one oscillator at w, the drive's own), the labels
sigma at the outputs, the step size normalisation (95th percentile of |dm|, a global gain), and the
conserved-total projection (subtract one constant).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import g8_phiworld_wires as G
from brainbox.phimedium import P0, step

T_REF, CAP = 0.05, 4.0              # echo loudness: x min(T_REF*AMP/|T|, CAP)  (an output amplifier that saturates)
T_SET, T_MEAS = 400.0, 60.0     # each phase waits 400 t for the previous phase to ring down (G9a)
BETA = 0.3


def run_phase(phi, old, t0, m, p, src_in_k, s_out, drive_input, clock2=True):
    """Advance the medium through one phase. Returns state, new clock time, output phasors (lock-in at w)
    and the material signal Q (lock-in of (phi - mean)^2 at 2w, or the mean of (phi - mean)^2 if not clock2)."""
    src_in, src_out, w_out = G.ports()
    dt, om = p["dt"], G.OMEGA
    period = 2 * np.pi / om
    n_set = int(round(T_SET / dt))
    n_meas = int(round(round(T_MEAS / period) * period / dt))
    pd = dict(p, damping=G.DAMP_MAP)
    spatial_c = np.einsum("bk,kyx->byx", np.real(s_out), src_out) if s_out is not None else 0.0   # Re part of source
    spatial_s = np.einsum("bk,kyx->byx", np.imag(s_out), src_out) if s_out is not None else 0.0
    B = phi.shape[0]
    S1 = np.zeros_like(phi); Sc = np.zeros_like(phi); Ss = np.zeros_like(phi)       # sum phi, phi cos2wt, phi sin2wt
    Q2c = np.zeros_like(phi); Q2s = np.zeros_like(phi); P2 = np.zeros_like(phi)    # phi^2 cos, sin, mean
    oc = np.zeros((B, G.K)); os_ = np.zeros((B, G.K)); o1 = np.zeros((B, G.K))
    c1s = s1s = c2s = s2s = 0.0                     # sums of the clock itself (exactly 0 only for whole periods)
    t = t0
    for n in range(n_set + n_meas):
        f = 0.0
        if drive_input:
            f = f + G.AMP * src_in_k * np.sin(om * t)
        if s_out is not None:                       # source phasor S: Im(S e^{iwt}) = Re S sin + Im S cos
            f = f + spatial_c * np.sin(om * t) + spatial_s * np.cos(om * t)
        phi, old = step(phi, old, m, pd, "gap", "neumann", f)
        t += dt
        if n >= n_set:
            c2, s2 = np.cos(2 * om * t), np.sin(2 * om * t)
            ph2 = phi * phi
            S1 += phi; Sc += phi * c2; Ss += phi * s2
            Q2c += ph2 * c2; Q2s += ph2 * s2; P2 += ph2
            ro = np.einsum("byx,jyx->bj", phi, w_out)
            c1, s1 = np.cos(om * t), np.sin(om * t)
            oc += ro * c1; os_ += ro * s1; o1 += ro
            c1s += c1; s1s += s1; c2s += c2; s2s += s2
    mu = S1 / n_meas
    ro_mu = o1 / n_meas
    T = 2 * ((os_ - ro_mu * s1s) + 1j * (oc - ro_mu * c1s)) / n_meas       # output phasors (B, K), mean removed
    if clock2:   # 2w lock-in of (phi - mu)^2 = lockin[phi^2] - 2 mu lockin[phi] + mu^2 lockin[1]
        Q = 2 * ((Q2s - 2 * mu * Ss + mu ** 2 * s2s) + 1j * (Q2c - 2 * mu * Sc + mu ** 2 * c2s)) / n_meas
    else:        # DC part of (phi - mu)^2: what a plain intensity sensor sees
        Q = P2 / n_meas - mu ** 2 + 0j
    return phi, old, t, T, Q


def physical_epoch(phi, old, t, m, p, pi, phases=3, clock2=True, beta=BETA):
    """One teaching epoch on a batch of independent media (B, NY, NX). Returns dm signal W."""
    src_in, src_out, _ = G.ports()
    B = phi.shape[0]
    W = np.zeros_like(phi)
    for k in range(G.K):
        phi, old, t, T, Qf = run_phase(phi, old, t, m, p, src_in[k], None, True, clock2)
        s = G.AMP * echo_weights(T, pi[k], beta)                                # (B, K), set at each output
        phi, old, t, _, Qn = run_phase(phi, old, t, m, p, src_in[k], s, True, clock2)
        if phases == 3:
            phi, old, t, _, Qe = run_phase(phi, old, t, m, p, src_in[k], s, False, clock2)
        else:
            Qe = 0.0
        D = Qn - Qf - Qe
        W += np.real(2j * D) if clock2 else np.real(D)
    return W, phi, old, t


def echo_weights(T, target, beta=BETA):
    """What output j re-emits (per unit drive amplitude), from its OWN received phasor T_j and its label:
        beta * sigma_j * min(T_REF*AMP/|T_j|, CAP) * conj(T_j)/|T_j|,  sigma = +1 target, -1/(K-1) others.
    Louder when it hears less: the gradient of log|T| (each input counts equally, as in G8b),
    capped like a saturating amplifier. T: (..., K) phasors in drive units."""
    a = np.abs(T) + 1e-30
    sigma = np.array([1.0 if j == target else -1.0 / (G.K - 1) for j in range(G.K)])
    gain = np.minimum(T_REF * G.AMP / a, CAP)
    return beta * sigma * gain * np.conj(T) / a


def analytic_term(phi, old, m, p, pi, beta=BETA):
    """G8b-style computed version of the same objective, from separate runs (for G9a only)."""
    Af, Ae, T = G.fields(phi, old, m, p)
    g = np.zeros_like(m)
    for k in range(G.K):
        s = echo_weights(T[k], pi[k], beta)            # Ae is already per AMP of drive
        for j in range(G.K):
            g += 2 * np.real(Af[k] * s[j] * Ae[j])
    return g


def born_batch(seeds, m, p):
    phis, olds = [], []
    for s in seeds:
        a, b = G.born(np.random.default_rng(s), m, p)
        phis.append(a); olds.append(b)
    return np.stack(phis), np.stack(olds)


def train_physical(seeds, p, pi_train, phases=3, clock2=True, epochs=G.EPOCHS, log_every=20, pi_test=G.PI_CYCLIC):
    B = len(seeds)
    m = np.full((B, G.NY, G.NX), G.M0)
    phi, old = born_batch(seeds, m[0], p)
    t = 0.0
    budget = G.M0 * G.NY * G.NX
    log = []
    for e in range(epochs):
        W, phi, old, t = physical_epoch(phi, old, t, m, p, pi_train, phases, clock2)
        for b in range(B):
            d = W[b]
            m[b] = G.project(m[b] + G.ETA * d / (np.percentile(np.abs(d), 95) + 1e-30), budget)
        phi, old = G.relax(phi, old, m, p)        # slow material changed; the field settles (no copy)
        if (e + 1) % log_every == 0:
            log.append((e + 1, [G.score(G.fields(phi[b], old[b], m[b], p, echo=False)[2], pi_test)["acc"] for b in range(B)]))
            print(f"    epoch {e + 1}: acc {log[-1][1]}", flush=True)
    return m, phi, old, log


def g9a(out):
    """Does the physical loop reproduce the computed term? Correlation over pixels, 3 media."""
    rows = []
    for form in ("variational", "phiworld"):
        p = dict(P0, form=form, kappa=5.0)
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            m = np.clip(G.M0 + 0.15 * rng.standard_normal((G.NY, G.NX)), 0, 1)
            phi, old = G.born(rng, m, p)
            G.T_SET = 150.0                                   # what G8b used
            ga150 = analytic_term(phi, old, m, p, G.PI_CYCLIC)
            G.T_SET = 600.0                                   # reference: computed from fully settled runs
            ga = analytic_term(phi, old, m, p, G.PI_CYCLIC)
            G.T_SET = 150.0
            r = dict(form=form, seed=seed, **{"computed with G8b's 150 t settle": float(np.corrcoef(ga150.ravel(), ga.ravel())[0, 1])})
            global T_SET
            for ts, name, ph, c2 in ((150.0, "3-phase, 2w clock, settle 150", 3, True), (400.0, "3-phase, 2w clock, settle 400", 3, True),
                                     (300.0, "3-phase, 2w clock, settle 300", 3, True), (400.0, "2-phase, 2w clock, settle 400", 2, True),
                                     (400.0, "3-phase, no clock (DC), settle 400", 3, False)):
                T_SET = ts
                W, _, _, _ = physical_epoch(phi[None].copy(), old[None].copy(), 0.0, m[None], p, G.PI_CYCLIC, ph, c2)
                r[name] = float(np.corrcoef(W[0].ravel(), ga.ravel())[0, 1])
            T_SET = 400.0
            rows.append(r)
            print(f"G9a {form:11s} seed {seed}: corr with computed term  " +
                  "  ".join(f"{k}: {v:+.3f}" for k, v in r.items() if k not in ("form", "seed")), flush=True)
    print("(reference = the computed term from separate runs settled 600 t)")
    json.dump(rows, open(os.path.join(out, "g9a_unit.json"), "w"), indent=1)


ARMS = [  # name, form, phases, clock2, train mapping
    ("physical 3-phase | variational k=5", "variational", 3, True, G.PI_CYCLIC),
    ("physical 2-phase | variational k=5", "variational", 2, True, G.PI_CYCLIC),
    ("no clock (DC intensity) | variational k=5", "variational", 3, False, G.PI_CYCLIC),
    ("physical 3-phase | his form k=5", "phiworld", 3, True, G.PI_CYCLIC),
    ("physical 3-phase | variational, WRONG labels", "variational", 3, True, G.PI_WRONG[G.PI_CYCLIC]),
]


def g9b_arm(ai):
    name, form, phases, c2, pi_train = ARMS[ai]
    p = dict(P0, form=form, kappa=5.0)
    t0 = time.time()
    print(f"[{name}] start", flush=True)
    m, phi, old, log = train_physical(G.SEEDS, p, pi_train, phases, c2)
    rows = []
    for b, seed in enumerate(G.SEEDS):
        T = G.fields(phi[b], old[b], m[b], p, echo=False)[2]
        r = dict(arm=name, seed=seed, final=G.score(T, G.PI_CYCLIC), T=np.abs(T / G.AMP).tolist(), log=log)
        P = np.abs(T)
        r["final"]["contrast"] = float(np.mean([P[k, G.PI_CYCLIC[k]] / np.mean([P[k, j] for j in range(G.K) if j != G.PI_CYCLIC[k]]) for k in range(G.K)]))
        if ai == 0:
            rng = np.random.default_rng(1000 + seed)
            ph2, o2 = G.born(rng, m[b], p)
            r["reborn"] = G.score(G.fields(ph2, o2, m[b], p, echo=False)[2], G.PI_CYCLIC)
            ms = rng.permutation(m[b].ravel()).reshape(m[b].shape)
            ph3, o3 = G.relax(phi[b], old[b], ms, p, 200.0)
            r["shuffled"] = G.score(G.fields(ph3, o3, ms, p, echo=False)[2], G.PI_CYCLIC)
            np.save(os.path.join(OUT, f"g9b_m_seed{seed}.npy"), np.stack([m[b], phi[b]]).astype(np.float32))
        rows.append(r)
        print(f"[{name}] seed {seed}: {r['final']}" + (f"  reborn {r['reborn']}  shuffled {r['shuffled']}" if "reborn" in r else ""), flush=True)
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
    return rows


OUT = os.path.join(os.path.dirname(__file__), "..", "results")

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("a", "all"):
        g9a(OUT)
    if what in ("b", "all"):
        from multiprocessing import Pool
        with Pool(2) as pool:
            parts = pool.map(g9b_arm, range(len(ARMS)), chunksize=1)
        rows = [r for part in parts for r in part]
        json.dump(dict(config=dict(T_SET=T_SET, T_MEAS=T_MEAS, BETA=BETA, EPOCHS=G.EPOCHS, ETA=G.ETA, SEEDS=G.SEEDS,
                                   PI=G.PI_CYCLIC), rows=rows), open(os.path.join(OUT, "g9b_learning.json"), "w"), indent=1)
        print("\nsummary: accuracy per seed | mean T_target | contrast")
        for a in ARMS:
            r = [x for x in rows if x["arm"] == a[0]]
            line = (f"{a[0]:46s} acc {np.mean([x['final']['acc'] for x in r]):.2f} {[round(x['final']['acc'], 2) for x in r]}"
                    f"  T {np.mean([x['final']['T_target'] for x in r]):.3f}  contrast {np.mean([x['final']['contrast'] for x in r]):.1f}")
            if "reborn" in r[0]:
                line += f"  | reborn {np.mean([x['reborn']['acc'] for x in r]):.2f}  shuffled {np.mean([x['shuffled']['acc'] for x in r]):.2f}"
            print(line)
