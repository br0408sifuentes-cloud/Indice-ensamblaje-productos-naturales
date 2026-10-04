import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
L = lambda f: pd.read_csv(f"datos/{f}.csv")
d = pd.concat([L("std_coconut").assign(g="nat"), L("std_npatlas").assign(g="nat"), L("std_chembl").assign(g="syn"),
               L("std_chembl_replica").assign(g="syn"), L("std_pubchem").assign(g="syn")]).dropna(subset=["MA_std_max", "frac_sp3"])
d = d[(d.enlaces_std >= 15) & (d.enlaces_std <= 35)].copy()
d["exceso"] = d.MA_std_max - d.enlaces_std.map(d.groupby("enlaces_std").MA_std_max.median())
fig = plt.figure(figsize=(3.45, 3.0)); gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
ax = fig.add_subplot(gs[1, 0]); axx = fig.add_subplot(gs[0, 0], sharex=ax); axy = fig.add_subplot(gs[1, 1], sharey=ax)
C = {"nat": "#2E7D32", "syn": "#6A1B9A"}; N = {"nat": "Naturales (COCONUT + NPAtlas)", "syn": "Sintéticos (ChEMBL + PubChem)"}
xx, yy = np.meshgrid(np.linspace(0, 1, 120), np.linspace(-8, 8, 120))
for g in ["syn", "nat"]:
    s = d[d.g == g]; k = gaussian_kde(np.vstack([s.frac_sp3, s.exceso + np.random.default_rng(0).uniform(-.3, .3, len(s))]), bw_method=.25)
    z = k(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape); lv = np.quantile(z[z > z.max() * .01], [.5, .75, .9])
    ax.contourf(xx, yy, z, levels=list(lv) + [z.max()], colors=[C[g]], alpha=.12); ax.contour(xx, yy, z, levels=lv, colors=[C[g]], linewidths=.9)
    ax.plot(s.frac_sp3.median(), s.exceso.median(), "o", ms=5, color=C[g], mec="white", mew=1.2, label=N[g], zorder=5)
    xs = np.linspace(0, 1, 200); kx = gaussian_kde(s.frac_sp3, .2)(xs); axx.fill_between(xs, kx, color=C[g], alpha=.2, lw=0); axx.plot(xs, kx, color=C[g], lw=1)
    ys = np.linspace(-8, 8, 200); ky = gaussian_kde(s.exceso, .3)(ys); axy.fill_betweenx(ys, ky, color=C[g], alpha=.2, lw=0); axy.plot(ky, ys, color=C[g], lw=1)
ax.axhline(0, color="k", lw=.6, ls=":")
ax.set_xlabel("Fsp3 (complejidad de forma) →"); ax.set_ylabel("MA respecto de la mediana de su tamaño →")
ax.set_xlim(0, 1); ax.set_ylim(-8, 8); ax.legend(loc="upper right", fontsize=5.8, handletextpad=0.2)
for a in (axx, axy): a.set_xticks([]) if a is axy else None; a.set_yticks([]) if a is axx else None
axx.tick_params(labelbottom=False); axy.tick_params(labelleft=False); axx.spines["left"].set_visible(False); axy.spines["bottom"].set_visible(False)
axy.set_xticks([]); axx.set_yticks([])
fig.savefig("figuras/Figura6_dos_familias_columna.png", dpi=300, bbox_inches="tight")
print(d.groupby("g")[["frac_sp3", "exceso"]].median())
