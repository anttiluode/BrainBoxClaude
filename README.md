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

### G2 — slow rhythm turns position into phase · **PASS** (the real finding of the day)
`gates/g2_pac.py` · 6 macro-blocks · soma also injects a slow sinusoid (pacemaker) · 5 seeds

- **Chain** of macro-blocks: the drive becomes a travelling phase wave, monotone along the chain,
  ≈ 25–30° per block, e.g. seed 0: −58°, −86°, −113°, −143°, 180°, 153°. Separation score
  (min block gap / within-block SD) **30.1**.
- **All-to-all** bridges: every block at the same phase, min gap **1.5°**, score **0.18**.

So bridge *topology* decides whether position can become phase. Nothing is labelled; the medium sets it.

Phase-gated release (each unit fires at the peak of its *local* slow oscillation — a PAC-like gate) vs
linear release, same read window: K=4 simultaneous units **1.00 gated vs 0.54 linear**; jitter up to 5 t
costs nothing. This part is close to guaranteed once phases differ (time-division multiplexing) and is
**not** claimed as a discovery — G2b prices it.

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

## Ledger — corrections made on the way (kept on purpose)

1. The morning scratch run reported 83% / 75% (macro / sub). It trained and tested the decoder on the same
   clean traces. Held-out: **58% / 31%**.
2. G1 first version let extra receivers *remove* candidate sub-blocks (R=8 had 8 atoms, R=1 had 15) — an
   easier problem dressed as a better receiver. Fixed: nested receivers, one fixed candidate list.
3. G2b first version included K = M = 6, where "all blocks active" is the only possible answer, so both
   arms scored perfectly. Removed.
4. Tier detector threshold 2.5 missed a 2.37× gap on one seed; set to 2.0, now [1,3,12,112] on all seeds.

## Prior art (subtract before claiming)

- Slow Laplacian eigenvectors as coordinates: spectral clustering, Laplacian eigenmaps.
- Spectral gaps as timescales of block structure: metastability, Markov state models.
- Connectivity → hierarchy of timescales in cortex: Chaudhuri et al., *Neuron* 2015; Murray et al. 2014.
- Phase as a position code from travelling theta along the hippocampal axis: Lubenov & Siapas 2009;
  Patel et al. 2012.
- Slow phase as a slot address (PAC): Lisman & Jensen, theta–gamma code.
- One-point read-out losing interior information: the Calderón problem / EIT.

What is not obviously in that list: the coupling of **two plasticities with two meanings** —
content written inside blocks, addresses written in the bridges — and that bridge topology (chain vs
all-to-all) is what decides whether a single pacemaker can turn the address into time. That is G3.

## What this means for dynamic AI (claim size: small)

- An address can come for free from weak coupling; it costs no parameters, but a single linear read-out
  only reaches ~#timescales dimensions of it.
- A slow rhythm through a *chain-like* bridge topology converts address into firing time, and in a band of
  rhythm frequencies that conversion pays for itself in throughput.
- Design rule to test next: **write memories inside blocks, keep bridges stable.** Bridge drift should
  scramble *where* things are while leaving *what* intact.

## Next gates

- **G3** — content vs address writes. Plasticity inside blocks vs on bridges; drift the bridges with and
  without a homeostatic rule; measure "address scrambled" vs "content intact" separately.
- **G4** — the honest discriminator: does a geometry-derived address beat a *learned* positional code in a
  matched-cost RNN / fast-weight model on a K-simultaneous recall task?

## Layout

```
brainbox/medium.py      medium builder (all / chain / flat), exact impulse and driven responses
gates/g0_tiers.py       G0
gates/g1_superposition.py
gates/g2_pac.py
gates/g2b_throughput.py
results/*.json          raw numbers for every gate
```

Requires numpy and scipy. Each gate runs in seconds on a laptop.
