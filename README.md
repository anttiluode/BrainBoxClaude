# BrainBoxClaude

**Can block geometry give a box of matter a free address system — and what does it cost to read it?**

Claude's half of the October 20, 2026 split (Sol builds the content / learning side; Argon observes).
Grew out of the morning's thread: the EPFL lipid atlas (539 "lipizones", *Nature* 2026,
DOI 10.1038/s41586-026-11050-0), the Mad Monday return-ping results in SihtiBrain, and the October 2024
"super weights summed into blocks" idea that started the whole dynamic-AI line.

> Do not hype. Do not lie. Just show.

## The idea under test

Fast nodes grouped into sub-blocks, sub-blocks into macro-blocks, each level joined by weaker
"bridge" conductances. A passive medium, `dV/dt = -L V + I`, solved exactly by eigendecomposition.

The claim: **weak bridges create slow modes, and the slow modes are an address.** A unit with the same
local state in a different place in the hierarchy produces a different soma trace — and the
difference sits in the *late* part of the trace:

```
h_i(t) = Σ_k φ_k(i) φ_k(soma) e^(-λ_k t)
```

At late times only the small-λ (slow) terms survive, and their coefficients φ_k(i) are block-constant —
digits of an address. No register stores them; they come from geometry.

The 2024 superweight idea ("sum weights so they work in blocks") is block-averaging, and the slow
eigenvectors are exactly block-constant. That naive idea was spectral coarse-graining.

## Gates and results (all numbers reproducible: `python gates/<gate>.py`, results in `results/`)

### G0 — tiers and single-unit address · **PASS** (modest)
`gates/g0_tiers.py` · 4 macro × 4 sub × 8 nodes = 128 nodes · 5 seeds · held-out nearest-centroid decoder

- Spectrum splits into tiers **[1, 3, 12, 112]** on every seed — exactly 1 global, M−1 macro, M(S−1) sub,
  rest local. Tier sizes are set by block counts, not matrix size.
- Slow eigenvectors are block digits: between-block / within-block SD of entries ≈ **40–57** (macro tier),
  **21–26** (sub tier).
- Same unit pulse, decode its location from the soma trace (noise 5% of RMS):

| window | macro | sub |
|---|---|---|
| fast (early, t < 2) | 0.47 | 0.20 |
| **slow (late, t ≥ 2)** | **0.58** | **0.31** |
| flat control (same edges, no hierarchy), slow | 0.25 | 0.07 |
| chance | 0.25 | 0.067 |

Early trace says *what*, late trace says *where*. The flat control sits at chance, so the address is geometry.
Decoding is modest — see G1 for why.

### G1 — superposition limit of one linear soma · **prediction CONFIRMED** (a designed failure)
`gates/g1_superposition.py` · K units in K distinct sub-blocks, unknown amplitudes U(0.5,1.5),
NNLS decode of the active set (K given to decoder), 8 candidate sub-blocks for every R

Pre-stated: one soma separates addresses by decay *rate*. Modes with near-equal λ are inseparable in
time (the 3 macro modes sit at λ ≈ 0.046, 0.049, 0.051), so resolvable address dimensions ≈ number of
distinct timescales, not number of modes. More receivers should restore it.

Exact-set recovery, noise 1%:

| receivers R | resolvable dims | K=1 | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|---|---|
| 1 | **3.8** | 0.76 | 0.29 | 0.11 | 0.06 | 0.06 |
| 2 | 4.8 | 0.91 | 0.55 | 0.26 | 0.12 | 0.10 |
| 4 | 7.0 | 0.89 | 0.78 | 0.60 | 0.50 | 0.42 |
| 8 | 7.2 | 0.95 | 0.84 | 0.72 | 0.61 | 0.57 |
| chance | | 0.125 | 0.036 | 0.018 | 0.014 | 0.018 |

Noiseless, the dictionary is full rank (8/8): **the information is present but not accessible at one point.**
This is the Calderón / "one soma is one boundary point" statement, measured.
Caveat: "noiseless" still includes within-block node variation (the decoder uses block centroids).

### G2 — slow rhythm turns position into phase · **PASS** for the phase map; gate = plain time-slotting
`gates/g2_pac.py` · 6 macro-blocks · soma also injects a slow sinusoid (pacemaker) · 5 seeds

- **Chain** of macro-blocks: the drive becomes a travelling phase wave, monotone along the chain,
  ≈ 25–30° per block, e.g. seed 0: −58°, −86°, −113°, −143°, 180°, 153°. Separation score
  (min block gap / within-block SD) **30.1**.
- **All-to-all** bridges: every block at the same phase, min gap **1.5°**, score **0.18**.

So bridge *topology* decides whether position can become phase. Nothing is labelled; the medium sets it.

Phase-gated release (each unit fires at the peak of its *local* slow oscillation — a PAC-like gate) vs
linear release, same read window: K=4 simultaneous units **1.00 gated vs 0.54 linear**.

**Fixed-slot control (added after Sol's BrainBoxSol PR #2 exposed the gap — my G2 originally lacked it).**
Release times assigned per block from an external label instead of from the medium:

| K = 4, release jitter | linear | phase-gated | fixed slots, same span as phase | fixed slots, whole period |
|---|---|---|---|---|
| 0 | 0.53 | 1.00 | 1.00 | 1.00 |
| 20 t | 0.53 | 0.86 | 0.88 | 1.00 |
| 50 t | 0.55 | 0.59 | 0.55 | **0.88** |

**The PAC-like gate is exactly as good as fixed time slots over the same span, and worse than evenly spaced
slots** (the medium's phase wave only covers ~210° of the cycle). The benefit is *temporal separation*, not
phase coupling. The only thing the phase gate adds is that **the medium assigns the slots itself, with no
labels** — and that assignment is only as stable as the bridges (G3, G4).

### G2b — does the gate buy throughput or just spend time? · **PASS in a band, FAIL outside it**
`gates/g2b_throughput.py` · units correctly addressed per unit time · 3 seeds

| drive ω | phase gap → time gap | gated K=4 (jit 0) | gated throughput (conservative) | linear throughput |
|---|---|---|---|---|
| 0.005 | 22° → 76 t | 1.00 | 0.0019–0.0020 (1.0–1.4×) | 0.0014–0.0019 |
| **0.01–0.02** | 33–47° → 40–60 t | 0.97–1.00 | **0.0024–0.0031 (1.35–1.9×)** | 0.0014–0.0019 |
| 0.04 | 41–46° → 18–20 t | 0.69–0.89 | 0.0021–0.0029 (~1.5×) | |
| ≥ 0.08 | collapses to 2–3 t | 0.37–0.51 | 0.0012–0.0019 (0.8–1.05×, no gain) | |

Conservative = full cycle **plus** full slow tail charged to the gated read. Release-time jitter of 15 t
costs 15–35 points in the good band — jitter matters relative to the time gap, not the period.
**My pre-stated criterion was wrong:** I said the time gap must exceed the soma's slow smear (~290 t);
gating already works at ~40 t because the decoder uses trace shape, not only separation.

### G3 — content lives in blocks, address lives in bridges · **PASS (double dissociation)**
`gates/g3_content_vs_address.py` · chain medium, 6 × 4 × 8 · 5 seeds · every read-out decoded with
**stale** templates learned before drift (a reader that re-learns nothing)

Content of a sub-block = its local fast fingerprint (pulse at its entry node, its own 8 nodes, t ≤ 3), read
locally. Address = (i) its soma late-trace signature (G0 read-out, sub-block level) and (ii) its macro-block's
phase under the pacemaker drive (G2 read-out). Drift = log-normal multiplicative noise exp(s·ξ) on ONE edge
class at a time. Pre-stated prediction: a double dissociation.

| drift (s = 1.0) | content ID | address ID (sub) | phase ID (macro) | phase shift |
|---|---|---|---|---|
| none (baseline) | 1.00 | 0.58 | 1.00 | 0° |
| **inside blocks** | **0.47** (0.17–0.67) | 0.60 (0.56–0.68) | 1.00 | 0.1° |
| **bridges** | 0.92 (0.88–0.96) | **0.18** (0.15–0.21) | **0.40** | 21° |

At s = 0.6: in-block → content 0.73, address 0.59 · bridge → content 0.98, address 0.25.
In-block drift does not move the address at all (centroid drift ratio 0.004); bridge drift leaves content
mostly intact (some leak at large drift: bridges also load the block). Ranges are min–max over seeds.

### G3b — which conserved quantity protects the address · **PASS (3 of 3 predictions)**
Same harness, bridge drift, then one homeostat:

| bridge drift s = 1.0 | address ID (sub) | phase ID (macro) | phase shift |
|---|---|---|---|
| none | 0.19 | 0.47 | 21.0° |
| **interface-total** (each sub-block-pair keeps its total bridge conductance) | **0.57** (0.48–0.62) | **1.00** | 0.1° |
| macro-interface (only macro-pair totals kept) | 0.37 | **1.00** | 0.7° |
| per-node synaptic scaling (classic; control) | 0.19 | 0.51 | 19.3° |

- **Interface totals are the conserved quantity.** Individual bridges can drift by a factor of e either way;
  as long as each interface keeps its total, the address survives. This is the correct version of this
  morning's "grid changes but the thing stays the same": the (approximate) nullspace of the address map is
  *redistribution inside an interface*.
- **Resolution follows the homeostat:** conserving only macro-pair totals protects the macro/phase address
  but not the sub-block address — the protected digit is exactly the tier whose totals are conserved.
- **Classic per-node scaling does nothing for the address** — bridges are too small a share of a node's total
  conductance for degree conservation to see them.
- Not perfect: at s = 1.0 the interface homeostat recovers 0.48–0.62 vs a 0.54–0.64 baseline.
  Redistribution inside an interface is invisible only to first order.
- Honest limit: the interface homeostat needs each block to sense its total coupling to each neighbour.
  That is local to the interface, not to a single synapse. Whether biology has such a sensor is not tested here.

### G4 — Sol's learned content under bridge drift, rescued by interface totals · **PASS (P1–P3); P4 fails at P = 1**
`gates/g4_sol_bridge_rescue.py` · `brainbox/solport.py` · source: anttiluode/BrainBoxSol
(branch `sol/local-material-credit-g0`, commit 973bb63)

Sol found that after his local adjoint writes, −20/+20/−20% bridge drift pushes read-out errors to ~15%,
and 160 supervised replay steps with **all** old targets still leave ~10–12%. Gemini/Argon proposed that
phase-gated addressing (G2) could protect his content. G3b says something else protects it: interface
totals. Tested on **Sol's own medium and writers**, ported here.

- **Reproduction first.** The port reproduces Sol's G0 exactly: local error 0.0015499, post-drift
  [+0.0402, −0.1467, −0.0000, −0.1457], replay [0.0000, −0.1165, −0.0001, −0.1037].
- **Sol's medium has one bridge edge per interface (P = 1).** There an "interface-total homeostat" just
  restores every bridge — it would have to store each weight. Degenerate; not counted as evidence.
- **The test:** P parallel bridges per interface, same total conductance as Sol's. Drift
  g ← g·exp(0.2·ξ_interface + 0.2·ξ_edge) (coherent + independent). The homeostat stores **one scalar per
  interface** and gets **no targets, no replay, no read-out**. Same drifted medium for every repair.

Sol's bridge conductance, all 384 cases (16 teacher patterns × 24 write orders), local writer, 3 drift draws:

| P | error before drift | after drift | Sol's replay (all targets) | **interface homeostat** | per-node scaling | homeostat beats replay |
|---|---|---|---|---|---|---|
| 1 (degenerate) | 0.0154 | 0.146 | 0.069 (68% removed) | 0.0154 (100%) | 0.042 (77%) | 76% |
| 2 | 0.0035 | 0.132 | 0.071 (51%) | **0.0056 (98%)** | 0.126 (4%) | 97% |
| 4 | 0.0041 | 0.120 | 0.067 (49%) | **0.0052 (99%)** | 0.107 (10%) | 97% |
| 8 | 0.0033 | 0.113 | 0.062 (50%) | **0.0039 (99%)** | 0.102 (8%) | 96% |

(% = fraction of the drift damage removed, averaged per draw.)

- **Purely independent per-edge drift** (s = 0.3, no shared interface part — all of it redistribution inside
  interfaces, the hard case): homeostat removes **94% / 95% / 96%** at P = 2 / 4 / 8; replay 55% / 61% / 75%.
- **Sol's protected (OGD/GPM) writer:** learns cleaner (0.0002 before drift) but drift damages it the same
  (0.119 at P = 4). Protecting old outputs *while writing* does not protect them from the medium moving
  *afterwards*. Homeostat removes 97%, replay 45%.
- **Bridge / interior conductance ratio** (P = 4): homeostat removes 98 → 98 → 97 → 95 → **91%** as the ratio
  goes 0.03 → 0.06 → 0.12 → 0.24 → 0.50; replay rises 39 → 77%. The part the homeostat cannot undo grows as
  blocks stop being well-mixed relative to their bridges — the address is a coarse-grained quantity, and
  the coarse-graining is only as good as the spectral gap.

Predictions (written before running): **P1** reproduction — pass. **P2** homeostat removes > 70% and beats
replay — pass (91–99% removed; beats replay in 88–99% of draws for every P ≥ 2 setting, 69% at ratio 0.5).
**P3** residual grows with bridge/interior ratio — pass. **P4** node scaling < 30% — pass for every P ≥ 2
(3–26%), **fail at P = 1** (73–77%): with one bridge per interface, scaling the bridge's endpoint nodes
partly restores that single bridge.

Honest limits:
- The homeostat holds a **stored set point per interface** — one remembered number. Cheaper than replaying
  every old target, but not free, and it needs to sense the total coupling across an interface.
- Sol's replay is restricted to *local* writes by his design (bridges are never written); a replay allowed to
  write bridges could also do better. The comparison is homeostat vs **his** repair rule.
- Only bridge drift is tested here. In-block drift damages content directly (G3), and an interface homeostat
  does nothing about that.

**On Gemini's synthesis:** the pairing (Sol's local writes + Claude's address side) was the right pairing, but
the mechanism was wrong. Phase-gated read-out does not protect against drift — under bridge drift the phase
wave itself moves (21° in G3). What protects Sol's content is (1) **redundant bridges**, so drift is mostly
redistribution inside an interface, and (2) **conserving each interface's total**. Sol's medium had neither
redundancy nor a conserved quantity, which is why its address scaffold looked brittle.

### G5a — the grown box: growth writes the address, but one point cannot read it · **PASS (E1–E4)**
`gates/g5a_grown_box.py` · growth by splitting: every generation each box splits in two; nodes separated at
generation g are coupled by c·r^(L−1−g)/N (older splits = outer = weaker). Depth 6, 64 leaves. This is an
**ultrametric** medium; a leaf's address is its split history (6 binary digits — a 2-adic number).

- **E1.** One tier of timescales per generation, sizes **1, 2, 4, 8, 16, 32**, and every tier's eigenspace is
  spanned exactly by that generation's Haar functions (overlap **1.000** for all tiers, both r). The medium's
  own modes *are* the block pyramid: each mode is "the difference between the two halves of one box".
- **E2.** Slowest timescale grows as r^−(L−1): **9.1·10⁴** at r = 0.1, **411** at r = 0.3 (fastest ~19–28).
  The outer box diverges in time — depth has to be capped by the range of timescales a reader can wait for.
- **E3 (predicted before checking).** Coupling depends only on ultrametric distance, so one soma sees only the
  *distance* to a source, not its address: **7 distinct traces for 64 nodes** (6 distances + itself).
  Random extra receivers help slowly: unique addresses 11% (R=1) → 18% → 27% → 41% → 62% (R=16).
- **E4.** Reading far (outer) distance digits needs an observation window of the outer timescale: at r = 0.1 the
  5th distance class goes 0.03 → 0.10 → 0.47 → 0.95 → 1.00 as the window grows 1 → 10⁴ (1% noise). (The farthest
  class is half of all nodes and wins near-zero ties at short windows; its early scores are not readability.)
- **Imperfect growth makes the address readable.** Full-address decode at one soma (63 candidates, chance 0.016):

| growth jitter | r = 0.3, noise 0.1% | r = 0.3, noise 1% | r = 0.1, noise 0.1% |
|---|---|---|---|
| 0 (perfect) | 0.10 | 0.11 | 0.07 |
| 0.05 | 0.91 | 0.31 | 0.32 |
| 0.2 | 1.00 | 0.61 | 0.45 |
| 0.5 | 1.00 | 0.89 | 0.61 |

Perfect symmetry hides the address from a single reader; disorder in growth reveals it (the same reason G0's
random block graphs were decodable). Prior art: ultrametric diffusion (Ogielski & Stein 1985), p-adic wavelets as
eigenvectors of ultrametric operators (Kozyrev), Avetisov–Bikulov–Kozyrev; ultrametricity in spin glasses
(Rammal, Toulouse & Virasoro 1986). Nothing here is new mathematics; the link to addressing is ours.

### G5b — do conserved block totals protect a trained classifier from drift? · **mechanism PASSES, as an AI fix FAILS**
`gates/g5b_classifier_drift.py` · sklearn digits (8×8), MLP 64→64→10 trained in numpy, test accuracy 0.975 ·
first-layer weights drift w ← w·exp(s·ξ) · repair restores each group's stored total (equal-share correction) ·
3 training seeds × 20 drift draws

Fraction of the drift damage removed (natural inputs):

| repair (stored numbers) | s = 0.3 | s = 0.6 | s = 1.0 |
|---|---|---|---|
| per-unit row total (64) | 29% | 18% | 4% |
| local 4×4 block totals (256) | 33% | 22% | −2% |
| **local 2×2 block totals (1024)** | **82%** | **76%** | **52%** |
| random groups of 4 (1024, no locality) | 53% | 38% | 10% |
| 2-bit snapshot of all weights (bits = 256 floats) | −101% | 58% | 87% |
| **8-bit snapshot of all weights (bits = 1024 floats)** | **100%** | **100%** | **100%** |

On **permuted** pixels (blocks of unrelated pixels) the local 2×2 advantage disappears: 22% / 27% / 2%.

- **Q1 pass:** at identical storage, *local* totals beat random groups (82 vs 53%, 76 vs 38%, 52 vs 10%).
- **Q2 pass:** the advantage needs locality in the input — gone on permuted pixels. Within-block pixel correlation
  is 0.26 for 2×2 blocks, 0.02 for random groups; this is the classifier version of G4's ratio law.
- **Q3 mostly fails:** 4×4 blocks behave like whole-row totals — correlation only lives at the 2×2 scale.
- **The kill:** a bit-matched 8-bit snapshot recovers everything, and even a 2-bit snapshot (¼ the storage) beats
  block totals at strong drift. **On digital hardware, keep a cheap copy.** Block-total conservation is not a
  competitive drift fix for ordinary AI.

Where it can still matter, stated as hypotheses, not results: substrates where weights cannot be read out and
copied — analog / in-memory hardware (prior art: global drift compensation in phase-change-memory inference,
Joshi et al., *Nature Communications* 2020 — our version is the local, block-wise refinement) and biology — and
the version that needs **no stored numbers at all**: plasticity that is conservative by construction
(Royer & Paré, *Nature* 2003). And in digital AI, random weight drift is not the real problem; new learning is.
The live test is continual learning, against OGD/GPM/EWC, not drift repair.

### G6 — the AI question: does conserved-pyramid plasticity help continual learning? · **FAIL (pre-stated rule)**
`gates/g6_continual.py` (v0) · `gates/g6b_continual_rotated.py` (the informative run)

Pre-stated rule: the pyramid counts as AI-relevant only if its final accuracy matches or beats **GPM at the same
protected rank** (Gradient Projection Memory, Saha et al. 2021 — learns and stores the protected subspace) while
storing nothing. Pyramid rule: after task 1, first-layer updates must be zero-sum inside every local 2×2 pixel block
(block totals conserved). Its protected subspace is the geometry; it stores no basis.

**v0 — split digits, one head per task: uninformative.** Plain fine-tuning already ends at 0.990 with forgetting
−0.007; every method lies within 0.006. The formal "FAIL" (pyramid − GPM-16 = −0.001) means nothing at that
ceiling. Kept for the record; redesigned before rerunning, rule unchanged.

**v1 — rotated digits, domain-incremental, one shared 10-way head** (tasks = all digits rotated 0/30/60/90/120°,
64→32→10, SGD+momentum, 5 seeds). Every projection arm protects the head identically (GPM on hidden activations),
so they differ **only** in the input-layer subspace: learned (GPM) vs geometric (pyramid) vs random.

| method | final ACC (±sd) | forgetting (BWT) | input basis stored |
|---|---|---|---|
| GPM (learned, rank grows to 29) | **0.736** (0.021) | −0.25 | 1856 numbers |
| EWC (λ = 10, best of grid; optimistic) | 0.716 (0.016) | −0.23 | Fisher + anchor, both layers |
| GPM-16 (learned, rank 16) | **0.553** (0.025) | −0.49 | 1024 |
| **pyramid 2×2 (geometric, rank 16)** | **0.486** (0.013) | −0.58 | **0** |
| pyramid 4×4 (rank 4) | 0.467 (0.012) | −0.63 | 0 |
| random groups (rank 16) | 0.451 (0.021) | −0.64 | 0 |
| fine-tune | 0.441 (0.003) | −0.66 | 0 |

- **Verdict: FAIL.** Pyramid − GPM-16 = **−0.067**; the pyramid loses on **5/5 seeds**
  (0.464–0.500 vs 0.524–0.593).
- What survives: locality is worth something — the pyramid beats random groups of the same rank on 4/5 seeds
  (0.486 vs 0.451) and beats fine-tuning. But a basis learned from the data at the same rank is clearly better, and
  standard methods (GPM, EWC) are far better still.
- Honest notes: EWC at λ ≥ 1000 diverged with this learning rate (overflow), so its grid effectively topped out at
  λ = 10; a tuned EWC could do better, which would not change the verdict. No hyperparameters were tuned for the
  pyramid or GPM arms.

**What G6 settles:** on digital hardware, where a network can store a basis or a weight copy, the brain-box
mechanisms (G5b: drift; G6: continual learning) lose to standard machinery. The fixed geometric subspace is a
data-blind approximation of what GPM learns. **The BrainBox line is about physical media** — substrates where
weights cannot be read out, bases cannot be computed and copies cannot be kept — **not about better digital AI.**

### G7a — does PhiWorld grow its own boxes? · **partly: it grows them, but they either churn or freeze into a few**
`gates/g7a_phiworld_spectrum.py` · headless port of his 2025 phiworld2 equations, copied exactly:
φ_tt = c²(φ)∇²φ − V′(φ) − γ∇⁴φ, c²(φ) = c₀/(1+κφ²) (mass slows the clock), V′ = −aφ + bφ³ (domains at ±√5),
defaults κ = 5, a = 1, b = 0.2, γ = 0.02, dt = 0.08, periodic grid. Start from his "random" initial noise.

**Mass-slows-the-clock prevents coarsening** (64×64, domains = connected regions with |φ| > 1, t = 1600):

| damping | κ = 5 (his rule) | κ = 0 (control) |
|---|---|---|
| 0.001 (his GUI default) | **663–726 domains**, still active | **1 domain** by t ≈ 160–800 |
| 0.003 | 1–3 | 1 |
| 0.01 | 2–3 | 1 |
| 0.03 | 4–7 (frozen) | 1 |
| 0.1 | 5–10 (frozen) | 1 |

**But the many domains are not persistent** (correction to my own first reading). At damping 0.001 only **52–56%** of
pixels keep their sign after 4 time units (0.50 = unrelated pattern): the "organic" field in the screenshots is a
churning state, not a set of boxes. Wherever domains *do* persist (sign kept 0.90–1.00 over 400 time units,
damping ≥ 0.003), there are only 1–8 of them. The window between damping 0.001 and 0.003 is not resolved yet.

Spectrum of the frozen states (48×48, relaxed to residual ~10⁻¹⁴, 3 seeds; no unstable modes):
- Modes below the uniform-domain band edge √2 exist only because of walls, and they are wall-enriched
  **5–8× relative to wall area** (κ = 5: 236–350 such modes, wall weight 0.14–0.16 vs wall area 0.02–0.04;
  κ = 0: 68 modes, weight 0.69 vs area 0.08).
- With κ = 5 the softest modes, which let walls drift, are stiffer: lowest frequency 0.86 (or 0.245 with one soft mode)
  vs 0.176 at κ = 0. Suggestive of wall pinning, but the frozen geometries differ between runs, so not matched.
- Note: PhiWorld is not a gradient system (c²(φ)∇²φ is not the derivative of any energy), so its frozen states
  depend on κ, unlike ordinary φ⁴ domains.

What this says for the box: the Clockfield rule *does* make the medium hold more than one block — the control
always melts to one — so it is the ingredient that lets PhiWorld build boxes at all. What it does not yet give is
**many blocks that stay put**, which is what an address system needs. Next: resolve damping 0.001–0.003, add
Sol's slow material field m (memory that outlives the fast field), and track domain identity, not just counts.

### G7b — the damping window G7a left open · **a sharp transition, and the hot state's "boxes" are not real**
`gates/g7b_damping_window.py` · his equations, 64×64 periodic, his noise init, 3 seeds

| damping | domains t = 800 → 1240 | sign kept after 4 / 40 / 440 t | rms velocity |
|---|---|---|---|
| 0.001 (his GUI default) | 668–683 → 681–722 | 0.54–0.58 / 0.49–0.50 / 0.49–0.50 | 3.5–3.6 |
| 0.0015 | 591–631 → 649–697 | 0.54–0.55 / ~0.50 / ~0.51 | 3.1 |
| 0.002 | 501–553 → 586–661 | 0.60 / ~0.50 / ~0.51 | 2.8–2.9 |
| **0.0025** | **37–134 → 40–279** | **0.91–0.98 / 0.85–0.96 / 0.75–0.93** | 1.0–1.9 |
| 0.003 | 8–24 → 2–10 | 0.98–0.99 / 0.96–0.98 / 0.90–0.91 | 0.6–0.75 |
| 0.005 | 3 → 3 | 1.00 / 0.99 / 0.87–0.99 | 0.17–0.27 |

- Between 0.002 and 0.0025 the field jumps from a hot, fully reshuffling state (sign memory gone within 40 t) to a
  slow one whose domains persist. 0.0025 sits right at the edge: seed-to-seed spread is large (37 vs 134 domains).
- **Most of the hot state's "domains" are fluctuations, not boxes.** Take the hot state (damping 0.0025, 48×72,
  263 / 124 / 63 domains) and quench it with strong damping: 6–12 / 3–8 / 2–3 survive. A cold PhiWorld at his
  parameters holds only a handful of domains.

### G8.0 — Grok's `brain_laplacian`, tested · **the code does not do what the description says**
`gates/g8_0_operator.py` · `brainbox/phimedium.py` (corrected operator)

Grok (October 2026) proposed giving PhiWorld a slow material field *m* and an anisotropic operator
∇·(σ(m,φ)∇φ), σ = σ₀I + m(α n nᵀ + β t tᵀ), with code. At m = 0 any such operator must equal the plain
Laplacian PhiWorld uses.

| check | true Laplacian | Grok's code | corrected (face fluxes) |
|---|---|---|---|
| Gaussian, value at centre | −0.123 | **+0.118** | −0.123 |
| correlation with the true Laplacian | 1 | **−0.9998** | 1.000 |
| checkerboard (finest grid mode) | 8 | **0** | 8 |
| on a straight wall (m = 1): conductance across / along | — | **9.0 / 1.3** | 1.3 / 9.0 |

- **Sign flipped:** the divergence is taken as f[i−1] − f[i+1]; the operator is an anti-Laplacian (backward diffusion).
- **Spacing-2 stencil:** centred differences applied twice skip neighbours and cannot see the checkerboard mode.
- **Wrong direction:** n = ∇φ/|∇φ| points *across* a wall, so α ≫ β on n nᵀ conducts across walls, not along them.
- Corrected version: forward differences on cell faces, face conductances, zero boundary flux. Equal to the 5-point
  Laplacian at σ = 1 (difference 4·10⁻¹⁵), conservative, self-adjoint. (Off-diagonal σ_xy dropped; exact for
  axis-aligned walls.)

### G8a — are walls wires? is an m-stripe a wire? · **domains insulate; m-stripes are wires; his walls are not**
`gates/g8a_wires.py` · 32×48 box, absorbing edges, damping 0.005 · drive the middle input, read the middle output
31 cells away (and the two neighbouring outputs = leak) · lock-in amplitude / drive amplitude ·
m couples through the gap: a → a(1 − m), so a domain's small-signal gap is √(2a(1 − m))

| medium | κ = 5 (his rule), best ω | leak | κ = 0, best ω | leak |
|---|---|---|---|---|
| one domain, m = 0 (gap √2) | **< 10⁻⁴** at every ω | 0 | ≤ 0.004 | ≤ 0.001 |
| one domain, m = 0.7 (the G8b start) | 0.016 (ω 0.8) | 0.015 | 1.27 (ω 0.8) | 0.69 |
| **a wall** (+ above, − below), m = 0 | **< 10⁻⁴** at every ω | 0 | **0.70** (ω 0.8) | 0.0001 |
| m = 1 stripe, width 2 | 0.045 (ω 1.2) | 0 | 0.59 (ω 1.2) | 0.005 |
| **m = 1 stripe, width 4** | **1.07** (ω 1.0) | **0** | 2.95 (ω 1.2) | 0.011 |
| m = 1 stripe, width 6 | 7.4 (ω 0.8, resonance) | 0 | 4.07 (ω 1.0) | 0.007 |

ω tested: 0.6, 0.8, 1.0, 1.2 (and for the κ = 5 wall also 0.2–1.7: all < 10⁻⁴). Values above 1 are resonances of the
finite stripe (its ends reflect), not gain.

- **P1 pass:** a bare domain carries nothing below its gap. With mass-slows-the-clock the insulation is total:
  every stripe has **zero** leak at κ = 5.
- **P2 fails for his rule.** At κ = 5 a wall is a **one-cell step** (+2.0 → −2.0 between neighbouring pixels): the wall
  width √(c²/2a) is ~0.15 cells because the clock is so slow inside domains. There is no wall interior to carry a
  signal. At κ = 0 walls are a few cells wide and are good wires (0.70, leak 10⁻⁴). So in PhiWorld as he wrote it,
  **Grok's "walls are highways" is false**: the highways have to be made of m.
- **P3 pass:** an m = 1 stripe ≥ 4 cells wide carries the signal end to end with zero leak to the neighbours.

### G8b — can PhiWorld wire itself? · **PASS with an outcome-gated writer in a reciprocal medium; every writer without an outcome signal stays at chance**
`gates/g8_phiworld_wires.py` (medium, writers) · `gates/g8b_learning.py` (runner) · `gates/g8b_figure.py` ·
`results/g8b_learning.json`, `results/g8b_wires.png`

Same box as G8a. Three inputs on the left, three outputs on the right. The mapping to learn is
in0→out1, in1→out2, in2→out0: in 2-D every route has to cross another one. Nothing is wired by hand.
The medium is only **driven**: forward runs (drive input k at ω = 1.0) and echo runs (drive output j, the
return ping). A slow field m (the gap closer from G8a) is written from those runs. Total m is conserved
(Σm = 0.7 × area): a writer can only move material, never add it. 60 epochs, the same step size for every
writer. Read-out: drive input k alone, measure all three outputs → T[k, j]. Accuracy = the strongest output is
the right one (chance 0.33).

Writers:
- **three-factor:** Δm(x) = Σ_kj w_kj · Re(conj(T_kj) · A_fwd,k(x) · A_echo,j(x)). Forward field × echo field at x
  (local, phase-sensitive) × **one outcome number per pair** (w = 2/|T_k,target|² for the right output,
  −1/Σ_j|T_kj|² for every output). If the medium is reciprocal and the fields are settled, this is the gradient of
  Σ_k [2 log|T_k,target|² − log Σ_j |T_kj|²]. **Correction (found in G9a):** the fields here were read after
  150 t, which is not settled; the same computed term correlates only −0.15 to 0.74 with its settled version (G9a). So G8b's writer
  was a rough, transient-contaminated version of the gradient. It still learned, and the controls still sat at
  chance, but "exact gradient" is not what ran.
- **coincidence (Hebbian):** Δm ∝ |A_fwd,k| · |A_echo,target(k)|, each pair writing the same total. This is the
  forward × echo rule I proposed before this gate. No outcome number.
- **forward only:** Δm ∝ |A_fwd,k|² (no echo). **grok:** Δm ∝ |∇φ| (all walls, no drive). **no learning.**

Media: his PhiWorld rule c²(φ)∇²φ (κ = 5), the **variational** form of the same rule (κ = 5, see G8c), and κ = 0.

| writer · medium | accuracy (3 seeds) | target transmission | target / other outputs |
|---|---|---|---|
| **three-factor · variational κ = 5** | **1.00** [1, 1, 1] | **0.18** | **4.0** |
| three-factor · variational κ = 5, uniform start (no PhiWorld birth) | 1.00 [1, 1, 1]* | 0.19 | 4.2 |
| three-factor · his form κ = 5 | 0.56 [1, 0.33, 0.33] | 0.05 | 1.7 |
| three-factor · κ = 0 | 1.00 [1, 1, 1]* | 2.07 | 6.5 |
| three-factor · variational κ = 5, trained on a **different** mapping | **0.00** [0, 0, 0] | 0.05 | 0.5 |
| coincidence (Hebbian) · variational κ = 5 | 0.33 [0.67, 0, 0.33] | 0.016 | 1.1 |
| coincidence (Hebbian) · his form κ = 5 | 0.11 [0, 0.33, 0] | 0.008 | 0.9 |
| forward only · variational κ = 5 | 0.33 [0, 0.67, 0.33] | 0.033 | 1.3 |
| grok \|∇φ\| · variational κ = 5 | 0.44 [0.33, 0.33, 0.67] | 0.027 | 2.4 |
| no learning | 0.33 | 0.005 | 1.3 |

\* κ = 0 and the uniform start are deterministic: every seed relaxes to the same single domain, so their three
"seeds" are one run repeated. The independent-seed evidence is the variational κ = 5 row.

**Is the memory in m?** Two tests on the trained three-factor media:

| | trained | φ re-born (new noise, hot phase, cooled; m kept) | m shuffled (same values, scrambled positions) |
|---|---|---|---|
| variational κ = 5 | 1.00 (T 0.18) | **0.89** [0.67, 1, 1] (T 0.14) | **0.22** (T 0.01) |
| his form κ = 5 | 0.56 | 0.78 | 0.56 (T 0.006, noise level) |
| κ = 0 | 1.00 (T 2.07) | 1.00 (T 1.98) | 0.78 (T 0.32) |

Pre-stated (after the prototypes in ledger 9–11, before this run):
- **Q1 pass:** three-factor in a reciprocal medium learns the crossing mapping on 3/3 independent seeds.
  Not a blind prediction: the prototypes had already shown it on seed 0.
- **Q2 pass:** his non-reciprocal form does worse with the same writer (0.56, one seed of three).
- **Q3 pass:** Hebbian coincidence (0.33 / 0.11), forward-only (0.33), Grok's rule (0.44) and no learning (0.33) all
  sit at chance. **Forward field × return ping is not enough.** What makes it work is the per-pair outcome number
  and the phase.
- **Q4 pass (variational):** a newly born φ keeps 0.89; scrambling m drops to 0.22; training on another mapping gives
  0.00. The mapping lives in where the material sits, and survives a complete rebirth of the fast field.
  Q4 is **only partial at κ = 0**: a shuffled m still routes 0.78 at a sixth of the transmission. Part of the
  κ = 0 result does not depend on where m sits; not resolved.

What it is and is not:
- **Not clean wires.** The written m is a diffuse, hologram-like texture with faint diagonal channels
  (`results/g8b_wires.png`). The learned target transmission is 0.18 against 1.07 for the hand-drawn m-stripe of
  G8a. The writer found *a* solution, not the wiring diagram a person would draw.
- **PhiWorld's domains are not needed.** Starting from one uniform domain works as well (1.00, T 0.19). The κ = 5
  clock makes the medium a better insulator (G8a) and a worse conductor: κ = 0 transmits 10× more.
- The outcome number is one global scalar per input–output pair, broadcast to every point (a third factor, like a
  neuromodulator). It is not local. Forward and echo runs are separate runs from the same state; doing both at once
  (e.g. at two frequencies) is not tested. K = 3, 2-D, one box size.
- Prior art: this is in-situ adjoint training of a wave medium (Hughes et al., *Optica* 2018; Hughes et al.,
  *Science Advances* 2019, "Wave physics as an analog recurrent neural network"), with a three-factor plasticity
  reading (Frémaux & Gerstner 2016) and physical "coupled learning" (Stern et al., *PRX* 2021). New here is only the
  medium: PhiWorld, and what its clock rule does to the echo (G8c).

### G8c — is the return ping the adjoint? · **not in his PhiWorld; yes in the variational form of the same rule**
`gates/g8c_reciprocity.py` · `brainbox/phimedium.py: clock_variational`

Echo-based learning uses the field radiated *from* an output as the sensitivity of that output *to* every point.
That is true only if transmission is reciprocal: T(k → j) = T(j → k). His rule φ_tt = c²(φ)∇²φ + … is not the
force of any energy (G7a), so around a non-uniform background its linearisation is not symmetric (asymmetry 0.11 of
the largest entry on a random 8×9 field). The **variational** version of "mass slows the clock" comes from the energy
½c²(φ)|∇φ|²: φ_tt = ∇·(c²(φ)∇φ) − ½c²′(φ)|∇φ|² − V′(φ) − γ∇⁴φ. On the grid: face-averaged c², exact discrete
gradient (checked against −∂E/∂φ to 2·10⁻⁹), symmetric linearisation (4·10⁻¹¹), identical to his rule at κ = 0.

Reciprocity error Σ|T_kj − T′_jk| / Σ|T_kj| (drive k read j vs drive j read k):

| m | his form κ = 5 | variational κ = 5 | κ = 0 |
|---|---|---|---|
| uniform (uniform background) | 0.000 | 0.000 | 0.000 |
| random (m0 + 0.2 noise) | **0.40** | 0.03 | 0.002 |
| trained (G8b, variational, seed 0) | 0.10 | 0.04 | 0.002 |
| trained (G8b, his form, seed 0) | 0.14 | 0.04 | 0.002 |

The variational residual (3–4%) is not zero. Most likely cause: finite drive amplitude (nonlinearity) and slow
settling at κ = 5; not checked further. **Mass-slows-the-clock as he wrote it breaks reciprocity, so in his
PhiWorld the return ping is not a correct teacher.** Writing the same idea as an energy restores it, and the
learning (G8b, 0.56 → 1.00).

### G9 — the teacher made physical · **PARTIAL: it learns most routes from inside the medium, not reliably all; the clock control separates in transmission, not clearly in accuracy**
`gates/g9_physical_teacher.py` · `results/g9a_unit.json`, `results/g9b_learning.json`

G8b computed its teacher outside the medium: separate forward and echo runs from a copied state, and one outcome
number per pair broadcast to every pixel. G9 removes both. **One medium, one clock, the field never copied.**
For each input k, three phases in a row:

1. **free:** drive input k. Each output measures its own received phasor T_kj (a lock-in at the output).
2. **nudged:** drive input k **and** every output re-emits at ω with source
   s_j = β · σ_j · min(T_ref/|T_kj|, cap) · conj(T_kj)/|T_kj|, σ = +1 for its label "target", −½ otherwise.
   The output's own measurement and its own label decide its echo: louder when it hears less, saturating.
3. **echo:** outputs re-emit s_j, input silent.

The material at x senses only its own activity δ = φ − ⟨φ⟩ and the clock at 2ω: Q(x) = the 2ω lock-in of δ².
It writes Δm ∝ Re(2i [Q_nudged − Q_free − Q_echo]). Why 2ω: for δ = Im(A e^{iωt}), the 2ω part of δ² is −(i/2)A²,
so the three-phase difference is 2·A_fwd·A_echo — the product of the two fields **without** a complex conjugate,
which is what the gradient needs. A plain intensity sensor (the DC part of δ²) sees A_fwd·conj(A_echo): the wrong
product (control "no clock"). With these echoes the update is the gradient of
Σ_k [log|T_k,target| − mean_other log|T_kj|].

**G9a — does the physical loop compute the gradient?** Correlation over pixels with the same term computed from
separate, fully settled runs (600 t), 6 media (3 variational, 3 his form), random m:

| | correlation |
|---|---|
| 3 phases, 2ω clock, each phase settles 400 t | **0.86 – 0.92** |
| same, 300 t | 0.45 – 0.90 |
| same, 150 t | −0.18 – 0.21 |
| 2 phases (no echo-only phase), 400 t | −0.10 – 0.27 |
| 3 phases, no clock (plain intensity), 400 t | 0.03 – 0.25 |
| the computed term itself at 150 t (what G8b did) | −0.15 – 0.74 |

The loop works only if each phase waits for the previous one to ring down (400 t ≈ 5000 steps). The echo-only
phase is needed: with loud echoes the β² self-term swamps the cross term.

**G9b — learning,** crossing mapping, 60 epochs, 3 seeds, all else as G8b:

| teacher · medium | accuracy on the taught mapping (chance 0.33) | target transmission |
|---|---|---|
| **physical, 3 phases · variational κ = 5** | **0.78** [0.67, 1.00, 0.67] | **0.19** |
| same, taught a different mapping (scored on that mapping) | **0.78** [1.00, 0.67, 0.67] — and 0.00 on the cyclic one | 0.03 on cyclic targets |
| physical, 2 phases | 0.56 [0.67, 0.67, 0.33] | 0.035 |
| no clock (plain intensity) | 0.56 [0.67, 0.33, 0.67] | 0.040 |
| physical, 3 phases · his form κ = 5 | 0.44 [0.33, 0.67, 0.33] | 0.038 |
| (G8b computed teacher, variational, for reference) | 1.00 [1, 1, 1] | 0.18 |

Memory, physical 3-phase: φ re-born from noise with m kept → 0.67 [0.67, 1.00, 0.33]; m shuffled → 0.11.

Pre-stated (v2, same statements as v1):
- **R1 fails as stated** (≥ 2/3 seeds perfect): 1/3 seeds perfect. Pooled over both taught mappings the physical
  teacher gets **14 of 18 routes right** (chance 6), 2 of 6 runs perfect. The labels steer it: teach the other mapping
  and it learns that one (7/9) and scores 0/9 on the cyclic one.
- **R2 fails as stated:** the no-clock writer is not at chance (5/9 routes). It does separate in **transmission**:
  the clocked teacher builds routes ~5× stronger (0.19 vs 0.04), as strong as G8b's computed teacher. With 3
  outputs and 3 seeds an accuracy difference of 7/9 vs 5/9 is not significant.
- **R3 mostly passes:** wrong labels → 0.00 on the cyclic mapping; shuffled m → 0.11; re-born φ keeps 0.67 (one seed
  drops to 0.33).
- R4 (open): 2-phase 0.56; his non-reciprocal form 0.44 — lowest of the physical arms, consistent with G8c.

What is still outside the medium, stated plainly: the clock (one oscillator at ω, the drive's own, used at 2ω by
every pixel); the labels at the outputs; the step-size normalisation (a global gain: the 95th percentile of |Δm|);
the conserved-total projection (subtract one constant); and the phase schedule (who drives when). Compared to G8b
what is gone: copied states, separate runs, and the per-pair number broadcast to every pixel — each output now
sets its own echo from its own measurement. Prior art: this is the two-phase / contrastive structure of
equilibrium propagation and coupled learning (Scellier & Bengio 2017; Stern et al. 2021) and the interference
reading of in-situ backpropagation (Hughes et al. 2018), here with a second-harmonic sensor instead of an
intensity sensor. The cost: one epoch = 9 phases × 460 t in a row, about 3× the simulation of a G8b epoch.

### G10a — hologram or wires? · **neither, cleanly: the signal rides wavelength-scale structure, but it is not a narrow-band Bragg grating**
`gates/g10a_hologram_or_wires.py` · `results/g10a_hologram_or_wires.json`

Gemini read `g8b_wires.png` as a volume Bragg grating that routes crossing channels by diffraction at
K = k_in − k_out. Testable. At ω = 1.0 in the m0 = 0.7 domain the wave has k ≈ 1.55 rad/cell (wavelength ≈ 4 cells), so
forward × echo fringes have periods of 2–4 cells. Split each trained m into **coarse** (Gaussian low-pass, σ = 2 cells)
and **fine** (the rest = the fringe band), rebuild media with the same total material, re-relax φ, read the 3 × 3
transmission. Pre-stated: hologram → fine-only keeps the routing, coarse-only loses it, and the routing is narrow-band
(Bragg width ≈ ±0.07 for a 31-cell crossing). Wires/refraction → the opposite.

| trained medium | fine-band share of m's variance | trained: acc · T | coarse only | fine only | ω range with acc 1.0 |
|---|---|---|---|---|---|
| G8b computed teacher, seed 0 | 0.77 | 1.00 · 0.125 | 1.00 · **0.007** | 0.33 · 0.145 | 0.85 – 1.10 |
| seed 1 | 0.81 | 1.00 · 0.185 | 0.33 · **0.008** | **1.00 · 0.191** | 0.85 – 1.05 |
| seed 2 | 0.83 | 1.00 · 0.147 | 0.67 · **0.010** | 0.67 · 0.171 | 0.90 – 1.00 |
| G9b physical teacher, seeds 0–2 | 0.88 – 0.90 | 0.67 / 1.00 / 0.67 · 0.16–0.19 | **0.002 – 0.006** | 0.67 / 0.67 / 0.67 · 0.07–0.11 | (only at 1.0, seed 1) |

(Accuracy at T < 0.01 is noise-level: the untrained medium is at 0.005.)

- **Not wires.** 77–90% of the learned structure is at the fringe scale. Remove it and the medium goes dark: transmission
  falls 15–80× (0.13–0.19 → 0.002–0.010), as dark as the untrained medium, even with the coarse part at twice its
  contrast. There are no smooth channels carrying the signal.
- **Not a clean Bragg grating either.** The fine part alone keeps the transmission (0.07–0.19) but the full routing on only
  1 of 3 G8b media, so the coarse part matters for which output wins. And the G8b routing holds over ω = 0.85–1.10,
  wider than a Bragg selectivity of ±0.07 would allow. With only ~8 wavelengths across the box, that is not surprising.
- Best description: a **wavelength-scale scattering medium**, like an inverse-designed photonic device. Gemini's
  "texture, not wires" is right; "Bragg diffraction at K = k_in − k_out" is not shown.

### G10b — canalization, and the real reason G8b was faint · **canalization: FAIL; the faintness was my unsettled gradient**
`gates/g10b_canalization.py` · `results/g10b_canalization.json`

Gemini's diagnosis of G8b's weak routes (target transmission 0.18 vs ~1.07 for a hand-drawn stripe): the gradient writer
"gets lazy" once the target wins, and nature fixes that with rich-get-richer flow (rivers, self-focusing). Test: the same
writer plus a conservative material flow up the forward-energy gradient,
J = χ m(1 − m) ∇E − D∇m, Δm = −∇·J (zero flux at the edges, so it needs no global projection). Every field is now read
after **400 t** of settling (G9a showed 150 t was not settled), so χ = 0 is also the first settled-gradient G8b.
Pre-stated: C1 χ > 0 raises transmission without losing accuracy; C2 (Gemini's number) T ≥ 0.8 with full accuracy —
my expectation: not reached.

| writer | accuracy [3 seeds] | target transmission [per seed] | φ re-born |
|---|---|---|---|
| G8b as run (fields at 150 t) | 1.00 [1, 1, 1] | 0.18 | 0.89 |
| **settled gradient, χ = 0** | 0.89 [1, 1, 0.67] | **0.63** [0.76, 0.89, 0.23] | 0.67 |
| + canalization χ = 0.05 (D = 0.05) | 0.89 [1, 1, 0.67] | 0.71 [0.71, 1.24, 0.18] | 0.56 |
| + canalization χ = 0.2 (D = 0.05) | 0.78 [0.67, 1, 0.67] | **0.03** | 0.67 |

- **The faint routes were my bug, not lazy learning.** With settled fields the same writer builds routes 3.5× stronger
  (0.63 mean; 0.76 and 0.89 on two seeds), within reach of the hand-drawn stripe. The price: less robust — one seed
  of three misses a route, and re-born φ keeps less (0.67 vs 0.89).
- **C1 fails.** χ = 0.05 changes nothing beyond seed scatter (0.71 vs 0.63); χ = 0.2 makes the medium dark (0.03).
  Where the material went: χ = 0.2 smoothed the fringes away (fine-band share of m's variance 0.55–0.59, vs 0.81–0.87
  at χ = 0), and G10a says the fringes carry the signal. Canalization fights the mechanism that works here.
- **C2:** one seed of three (χ = 0.05, seed 1: T 1.24, accuracy 1.0). Not reliably.
- Design flaw, stated: the χ > 0 arms also carry the diffusion D, the χ = 0 arm does not. Part of the smoothing may be D
  alone; no D-only control was run.

## Ledger — corrections made on the way (kept on purpose)

1. The morning scratch run reported 83% / 75% (macro / sub). It trained and tested the decoder on the same
   clean traces. Held-out: **58% / 31%**.
2. G1 first version let extra receivers *remove* candidate sub-blocks (R=8 had 8 atoms, R=1 had 15) — an
   easier problem dressed as a better receiver. Fixed: nested receivers, one fixed candidate list.
3. G2b first version included K = M = 6, where "all blocks active" is the only possible answer, so both
   arms scored perfectly. Removed.
4. Tier detector threshold 2.5 missed a 2.37× gap on one seed; set to 2.0, now [1,3,12,112] on all seeds.
5. G2 had no fixed-slot control, so it let "phase gating works" read as "PAC works". Sol's BrainBoxSol PR #2
   ran that control on his side; added here. Phase gating = fixed slots of the same span (see G2).
6. G5b's first run had no bit-matched control and made block totals look like a useful drift fix (76% of damage
   removed). An 8-bit snapshot with the same bit budget removes 100%. Control added; verdict reversed for digital AI.
7. G6 v0 used a benchmark with no forgetting to fix (fine-tune 0.990). Redesigned to rotated-digits domain-IL
   after seeing v0; the verdict rule was kept unchanged and the redesign was written before rerunning.
8. G7a first pass: the relaxed states had coarsened to stripes and the largest-gap "tier" detector was too crude;
   replaced by sub-gap wall-bound mode counts. Then "~700 domains at his GUI damping" first read as persistent boxes;
   a sign-persistence check showed they reshuffle within 4 time units. Both corrections are in G7a.

9. G8 lock-in, first version: it multiplied the *static* background field by sin/cos over a non-whole number of
   periods, giving a fake uniform "transmission" of 0.13 everywhere, identical for every medium. Fixed: subtract the
   starting state, integrate over whole periods.
10. G8 writer prototypes (scratch, not in the gate): pure intensity Hebbian in a reflecting box with m0 = 0.9 built
    **one hub joining all six ports** (a soma, not a switchboard), for crossing and non-crossing mappings alike.
    Per-pair normalisation and a contrastive term did not fix it; shrinking the material budget made the medium go
    dark before any route formed. The gate's settings (absorbing edges, m0 = 0.7, ω = 1.0, conserved total, no pruning)
    were chosen after these failures.
11. The three-factor sign. Two finite-difference checks (log of transmissions ~10⁻⁵) pointed to −g; trained with −g
    the medium learned the **inverse** mapping (κ = 0: accuracy 0, every input sent to the wrong neighbour).
    The derived sign (+g; a relaxed domain's gap closes with m, δV″ = −2a δm) learns. The finite-difference checks
    were too noisy to trust; the gate uses the derived sign.
12. G8b's predictions were written after the prototypes had shown the variational and κ = 0 arms learning on seed 0,
    so Q1 is a replication across seeds, not a blind prediction.
13. G8b's lock-in fields were read after only 150 t of settling. G9a compared the computed term at 150 t with the
    same term from fully settled runs (600 t): correlation −0.15 to 0.74 across six media. G8b learned with an
    approximate gradient, not the exact one it described. G9 uses 400 t per phase (0.86–0.92).
14. G9b v1 (echo of fixed loudness, sign ±1 = amplitude objective) failed its R1: 0.33 / 0.33 / 0.67. The transmission
    matrices showed why: one pair took all the material (seed 1: in1→out2 at 0.54, everything else ≈ 0.01). Changed
    to echoes that get louder when the output hears less (log objective, as in G8b), predictions unchanged, rerun as v2.
    v1 log kept: `results/g9b_v1_amplitude_objective.log`.
15. G8b's "faint hologram" (target transmission 0.18) was blamed — by Gemini, and by me as "the writer found a faint
    solution" — on the learning rule. G10b: with settled fields the same rule reaches 0.63 (0.76 / 0.89 on two seeds).
    The faintness came from the unsettled-field bug of ledger 13.

## Prior art (subtract before claiming)

- Slow Laplacian eigenvectors as coordinates: spectral clustering, Laplacian eigenmaps.
- Spectral gaps as timescales of block structure: metastability, Markov state models.
- Connectivity → hierarchy of timescales in cortex: Chaudhuri et al., *Neuron* 2015; Murray et al. 2014.
- Phase as a position code from travelling theta along the hippocampal axis: Lubenov & Siapas 2009;
  Patel et al. 2012.
- Slow phase as a slot address (PAC): Lisman & Jensen, theta–gamma code.
- One-point read-out losing interior information: the Calderón problem / EIT.
- Training a wave medium's material by forward × adjoint fields: Hughes, Minkov, Williamson & Fan, *Optica* 2018
  (in-situ backpropagation); Hughes, Williamson, Minkov & Fan, *Science Advances* 2019 (wave physics as an RNN).
- Local learning in physical networks: coupled learning (Stern et al., *PRX* 2021); equilibrium propagation
  (Scellier & Bengio 2017); three-factor plasticity (Frémaux & Gerstner 2016).
- Self-organising transport networks from use: *Physarum* (Tero et al., *Science* 2010).

What is not obviously in that list: **two plasticities with two meanings** in one passive medium —
content written inside blocks, addresses written in the bridges (G3), the *interface total* as the
conserved quantity that protects addresses while classic per-node scaling does not (G3b), and bridge
topology (chain vs all-to-all) deciding whether a single pacemaker can turn the address into time (G2).
Synaptic scaling, homeostatic plasticity and coarse-graining are all old; this specific split has not
been checked against the literature beyond this list — treat as unverified novelty.

## What this means for dynamic AI (claim size: small)

- An address can come for free from weak coupling; it costs no parameters, but a single linear read-out
  only reaches ~#timescales dimensions of it.
- A slow rhythm through a *chain-like* bridge topology converts address into firing time, and in a band of
  rhythm frequencies that conversion pays for itself in throughput.
- Design rule, now measured (G3): **write memories inside blocks, keep bridges stable.** Bridge drift
  scrambles *where* things are while leaving *what* intact, and vice versa.
- Stability rule (G3b, G4): protect **interface totals**, not individual weights and not per-unit totals.
  Confirmed on Sol's medium and writers (G4: 94–99% of drift damage removed, no targets).
- Growth writes addresses (G5a): split-growth gives one timescale tier per generation and Haar modes, but a
  perfectly symmetric box hides the address from a single reader; imperfect growth reveals it.
- **Limit found (G5b):** in a trained digital classifier the mechanism transfers (local totals beat random
  groups, and only when inputs are locally correlated), but a bit-matched weight snapshot beats it outright.
  As a drift fix for ordinary AI it loses.
- **Limit found (G6):** in continual learning the geometric pyramid (0.486) loses to equal-rank learned GPM (0.553)
  on every seed, and far behind full GPM (0.736) / EWC (0.716). It beats random groups — locality is real — but
  not learned subspaces.
- **Bottom line after G0–G6:** the measured mechanisms (free addresses from weak coupling, interface totals as the
  conserved quantity, growth writing addresses) are real **physical-media** results. As digital-AI methods they lose
  to cheap standard controls. Claim size for AI: none yet.
- **PhiWorld, G7b–G8c:** his field at his parameters makes only a handful of real domains once it cools, and its walls
  are one pixel thick, so they cannot carry a signal. Domains are excellent insulators. A slow material field that
  closes the gap makes wires (G8a). Driving the medium plus a return ping plus **one outcome number per input–output
  pair** writes a routing that crosses itself, on 3/3 seeds, and the routing survives a full rebirth of the fast
  field (G8b). Forward × echo alone — the rule proposed before the test — does not: chance. The echo is a correct
  teacher only if the medium is reciprocal; his clock rule is not, its variational form is (G8c).
  Claim size: a 3 × 3 routing in a 32 × 48 box. A self-wiring medium, not a brain.
- **G9:** the teacher can be moved into the medium — one clock, outputs that echo from their own measurement, a
  material that senses the 2ω part of its own activity — and it computes the gradient (0.86–0.92) if every phase
  is given time to ring down. Learning from it is weaker than from the computed teacher: 14/18 routes, 2/6 perfect.
- **G10:** the learned medium is neither wires nor a clean Bragg grating: a wavelength-scale scattering texture carries
  the signal (remove the fringes and it goes dark). With the gradient read correctly the routes are 3.5× stronger
  (0.63); rich-get-richer canalization does not help and, strong, smooths the fringes away and kills transmission.

## Next gates

- **G9 — make the teacher physical.** Done, partial (above). Open: why it learns 2 of 3 routes where the computed
  teacher learns 3 — settling (0.86–0.92, not 1), the echo cap, or too few epochs; more outputs and seeds so that
  accuracy differences between teachers can be resolved at all; and removing the global gain and the projection.
- **G10 — scale and crossing.** K = 3 in 2-D is a toy. Does the conserved-material writer reach K = 8? Crossing
  routes leak in 2-D; does a third dimension (or a folded sheet) remove that cost?
- **Against a baseline.** The same writer in a plain linear gapped medium (no Mexican hat) — does PhiWorld's
  non-linearity add anything to G8b, or only the insulation?

## Layout

```
brainbox/medium.py      medium builder (all / chain / flat), exact impulse and driven responses
gates/g0_tiers.py       G0
gates/g1_superposition.py
gates/g2_pac.py
gates/g2b_throughput.py
gates/g3_content_vs_address.py   G3 + G3b
brainbox/solport.py     port of BrainBoxSol's medium and writers (P parallel bridges)
gates/g4_sol_bridge_rescue.py    G4
gates/g5a_grown_box.py           G5a
gates/g5b_classifier_drift.py    G5b
gates/g6_continual.py            G6 v0 (ceiling, kept for the record)
gates/g6b_continual_rotated.py   G6 (the informative run)
gates/g7a_phiworld_spectrum.py   G7a (PhiWorld port, spectrum, coarsening, persistence)
gates/g7b_damping_window.py      G7b
brainbox/phimedium.py   PhiWorld + slow field m: his equations, the variational clock, face-flux operators, lock-in
gates/g8_0_operator.py           G8.0 (Grok's operator vs the corrected one)
gates/g8a_wires.py               G8a
gates/g8_phiworld_wires.py       G8b medium, ports, writers, training loop
gates/g8b_learning.py            G8b runner (10 arms x 3 seeds, ~45 min on 2 cores) · g8b_figure.py
gates/g8c_reciprocity.py         G8c
gates/g9_physical_teacher.py     G9a (python ... a) and G9b (python ... b, 5 arms x 3 seeds, ~2 h on 2 cores)
gates/g10a_hologram_or_wires.py  G10a (coarse / fine decomposition of the saved media, omega sweep)
gates/g10b_canalization.py       G10b (settled gradient + conservative material flow, ~40 min on 2 cores)
results/*.json          raw numbers for every gate
```

Requires numpy and scipy (matplotlib for the figure). G0–G7a run in seconds; G7b and G8 gates take minutes.
