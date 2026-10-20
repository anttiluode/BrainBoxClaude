"""Figure for G8b: the m that the three-factor writer left behind (seed 0), and the field phi."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import g8_phiworld_wires as G

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ARMS = [(0, "his form, kappa = 5"), (1, "variational, kappa = 5"), (2, "kappa = 0")]

fig, ax = plt.subplots(2, 3, figsize=(13, 5.6), constrained_layout=True)
for c, (ai, title) in enumerate(ARMS):
    m, phi = np.load(os.path.join(OUT, f"g8b_m_arm{ai}_seed0.npy"))
    im0 = ax[0, c].imshow(m, vmin=0, vmax=1, cmap="viridis")
    im1 = ax[1, c].imshow(phi, vmin=-2.3, vmax=2.3, cmap="RdBu_r")
    ax[0, c].set_title(f"m  ({title})", fontsize=10)
    ax[1, c].set_title("phi", fontsize=10)
    for a in ax[:, c]:
        for k, y in enumerate(G.PORT_Y):
            a.plot(G.X_IN, y, "o", mfc="white", mec="k", ms=7)
            a.text(G.X_IN - 5, y + 1, f"in {k}", fontsize=7)
            a.plot(G.X_OUT, y, "s", mfc="white", mec="k", ms=7)
            a.text(G.X_OUT + 2, y + 1, f"out {k}", fontsize=7)
        a.set_xticks([]); a.set_yticks([])
fig.colorbar(im0, ax=ax[0, :], shrink=0.8, label="m (1 = gap closed)")
fig.colorbar(im1, ax=ax[1, :], shrink=0.8, label="phi")
fig.suptitle("G8b: material written by forward x echo x outcome, mapping in0->out1, in1->out2, in2->out0 (seed 0)", fontsize=11)
plt.savefig(os.path.join(OUT, "g8b_wires.png"), dpi=90)
print("saved results/g8b_wires.png")
