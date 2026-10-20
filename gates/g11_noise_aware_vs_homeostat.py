"""G11 - NCM's corruption trick vs the interface homeostat, on Sol's medium and writer.

NCM (arXiv 2610.14162) trains its generator with some coarse codes deliberately replaced by wrong ones, so the
finer levels learn to cope with errors in the levels above. Gemini (Oct 20) proposed the same for BrainBoxSol's
bridge-drift failure: drift the bridges WHILE writing, so the content is written to survive drift.
That is noise-aware training (the analog-hardware version: Joshi et al., Nat. Commun. 2020).

Sol's medium and local writer, unchanged (brainbox/solport.py, reproduced exactly in G4). The noise-aware writer
is Sol's write_one with ONE change: each step reads the response and the sensitivity on a freshly drifted copy
of the medium (bridges g <- g exp(s_c xi_interface + s_i xi_edge), training drift s_c = s_i = 0.2); the update
goes to the real medium. Same number of steps, same evaluations per step, bridges never written.

v1 (results/g11_v1_naive_noise_aware.log): one drifted copy per step. Sol's step is a full normalised correction,
so it fits the LAST random drift: clean error x15 and no drift protection - a strawman of the trick. v2 (default):
each step averages the correction over M = 16 drifted copies (16x the reads of the clean writer, stated).

Then the real drift (fresh draws, never seen in training) and four arms:
    clean writer          | no repair        clean writer          | interface homeostat
    noise-aware writer    | no repair        noise-aware writer    | interface homeostat
Homeostat = G4's: each interface rescales its bridges to its stored total (one number per interface).
P = 1 (Sol's medium) makes the homeostat degenerate (it restores the single bridge exactly): reported, flagged.

Pre-stated (written before running):
  N1 noise-aware writing removes some drift damage with no repair (> 0%), but less than the homeostat at P >= 2.
  N2 noise-aware + homeostat is not worse than the homeostat alone.
  N3 noise-aware writing costs clean (pre-drift) accuracy.
  N4 under drift twice the training strength (0.4) the noise-aware gain shrinks; the homeostat still works.
  N5 (Sol's own medium, P = 1, the case that started it) noise-aware writing gives a modest gain only.
"""
import itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brainbox.solport import SolMedium, teacher_medium, train, repair_interface_total
from g4_sol_bridge_rescue import err, drift, check_reproduction, PATTERNS, ORDERS

S_TRAIN = 0.2
NOISE_WRITER = os.environ.get("G11_WRITER", "averaged")   # v1 ran "naive" (results/g11_v1_naive_noise_aware.log)


def write_noisy(box, state, block, target, baseline, rng, s=S_TRAIN, rate=.15):
    """Sol's write_one, but the read happens on a drifted copy of the medium."""
    noisy = drift(box, state, s, s, rng)
    response, sens = box.read(noisy, block)
    residual = (response - target[block]) / baseline[block]
    grad = sens / baseline[block]
    w = box.owner == block
    n2 = np.dot(grad[w], grad[w])
    state[w] -= rate * residual * grad[w] / max(n2, 1e-12)
    state[w] = np.clip(state[w], .1, 2.)


def write_averaged(box, state, block, target, baseline, rng, M=16, s=S_TRAIN, rate=.15):
    """Fair version: the step is Sol's normalised correction averaged over M drifted copies
    (a step on the EXPECTED error under drift). Costs M reads per step."""
    w = box.owner == block
    upd = np.zeros(w.sum())
    for _ in range(M):
        response, sens = box.read(drift(box, state, s, s, rng), block)
        residual = (response - target[block]) / baseline[block]
        grad = sens[w] / baseline[block]
        upd += residual * grad / max(np.dot(grad, grad), 1e-12)
    state[w] -= rate * upd / M
    state[w] = np.clip(state[w], .1, 2.)


def train_noisy(box, target, baseline, order, rng, steps=80):
    state = box.initial.copy()
    step = write_averaged if NOISE_WRITER == "averaged" else write_noisy
    for k in order:
        for _ in range(steps):
            step(box, state, k, target, baseline, rng)
    return state


def block(P, s_test, cases, draws, seed):
    rng = np.random.default_rng(seed)
    box = SolMedium(P=P)
    base = box.values(box.initial)
    rows = []
    for signs, order in cases:
        tgt = box.values(teacher_medium(box, signs))
        gc = train(box, tgt, base, order, "local")
        gn = train_noisy(box, tgt, base, order, rng)
        for _ in range(draws):
            xi_seed = rng.integers(1 << 30)
            # the SAME drift draw for both writers: same multiplicative factors on the bridges
            dc = drift(box, gc, s_test, s_test, np.random.default_rng(xi_seed))
            dn = drift(box, gn, s_test, s_test, np.random.default_rng(xi_seed))
            rows.append(dict(pre_clean=err(box, gc, tgt, base), pre_noisy=err(box, gn, tgt, base),
                             clean_none=err(box, dc, tgt, base), noisy_none=err(box, dn, tgt, base),
                             clean_homeo=err(box, repair_interface_total(box, dc, gc), tgt, base),
                             noisy_homeo=err(box, repair_interface_total(box, dn, gn), tgt, base)))
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    dmg = [r for r in rows if r["clean_none"] - r["pre_clean"] > 1e-4]
    # damage removed, relative to the CLEAN writer's undrifted error and its unrepaired drifted error
    for k in ("noisy_none", "clean_homeo", "noisy_homeo"):
        out[f"removed_{k}"] = float(np.mean([(r["clean_none"] - r[k]) / (r["clean_none"] - r["pre_clean"]) for r in dmg]))
    out["noisy_none_beats_clean_none"] = float(np.mean([r["noisy_none"] < r["clean_none"] for r in rows]))
    out["n"] = len(rows)
    return out


def run():
    res = dict(reproduction=check_reproduction(), s_train=S_TRAIN, noise_writer=NOISE_WRITER)
    cases = [(p, o) for p in PATTERNS for o in ORDERS]          # all 384
    res["rows"] = {}
    print("\nerrors = mean |relative read-out error| over 384 cases x 3 drift draws; 'removed' = share of the clean "
          "writer's drift damage removed")
    for s_test in (0.2, 0.4):
        for P in (1, 2, 4, 8):
            d = block(P, s_test, cases, 3, seed=100 + 10 * P + int(10 * s_test))
            res["rows"][f"P{P}_s{s_test}"] = d
            print(f"drift {s_test} P={P}{' (homeostat degenerate)' if P == 1 else ''}: "
                  f"pre clean {d['pre_clean']:.4f} noisy {d['pre_noisy']:.4f} | drifted, no repair: clean {d['clean_none']:.4f} "
                  f"noisy {d['noisy_none']:.4f} | homeostat: clean {d['clean_homeo']:.4f} noisy {d['noisy_homeo']:.4f} | removed: "
                  f"noise-aware {d['removed_noisy_none']:+.2f}  homeostat {d['removed_clean_homeo']:+.2f}  both "
                  f"{d['removed_noisy_homeo']:+.2f} | noise-aware wins per draw {d['noisy_none_beats_clean_none']:.2f}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "g11_noise_aware_vs_homeostat.json"), "w"), indent=1)


if __name__ == "__main__":
    run()
