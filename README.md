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

## Prior art (subtract before claiming)

- Slow Laplacian eigenvectors as coordinates: spectral clustering, Laplacian eigenmaps.
- Spectral gaps as timescales of block structure: metastability, Markov state models.
- Connectivity → hierarchy of timescales in cortex: Chaudhuri et al., *Neuron* 2015; Murray et al. 2014.
- Phase as a position code from travelling theta along the hippocampal axis: Lubenov & Siapas 2009;
  Patel et al. 2012.
- Slow phase as a slot address (PAC): Lisman & Jensen, theta–gamma code.
- One-point read-out losing interior information: the Calderón problem / EIT.

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
- Stability rule (G3b): protect **interface totals**, not individual weights and not per-unit totals.
  For a dynamic AI this suggests a cheap guard: let block-to-block weights wander, but renormalize each
  block-pair's total coupling. That is a testable continual-learning rule, not yet tested on a learning task.

## Next gates

- **G5** — the honest discriminator for AI: put the interface-total rule into a *learning* network
  (block-structured RNN / fast weights) on a continual-learning task with weight drift, against a
  matched-cost unstructured network with ordinary normalization. If it does not win there, everything here
  is a property of passive media, not an AI improvement.

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
results/*.json          raw numbers for every gate
```

Requires numpy and scipy. Each gate runs in seconds on a laptop.
