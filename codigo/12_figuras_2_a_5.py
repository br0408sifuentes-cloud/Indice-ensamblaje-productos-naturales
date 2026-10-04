# Figuras del articulo en espanol y con tipografia Times (version 4). Requiere solo el Paso 6 (CARPETA).
# Genera Figura2 a Figura5 en CARPETA/figuras_finales (PNG a 300 dpi y PDF). La Figura4 incluye el panel de rutas biologicas.
import subprocess, matplotlib.font_manager as fm
subprocess.run("apt-get -qq install -y fonts-liberation > /dev/null 2>&1", shell=True)
for _f in fm.findSystemFonts():
    try: fm.fontManager.addfont(_f)
    except Exception: pass
import os, json, warnings, io, re
import numpy as np, pandas as pd
import matplotlib; pass
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
CARPETA = globals().get("CARPETA", ".")
SAL = f"{CARPETA}/figuras_finales"; os.makedirs(SAL, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7, "legend.frameon": False})
C = {"nat": "#2E7D32", "nat2": "#81C784", "chembl": "#6A1B9A", "pubchem": "#1E88E5", "abio": "#E65100", "gris": "#9E9E9E"}
def letra(ax, t, x=-0.12, y=1.04): ax.text(x, y, t, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
def guardar(fig, n):
    fig.savefig(f"{SAL}/{n}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{SAL}/{n}.pdf", bbox_inches="tight"); print("Guardada:", n); plt.show()
L = lambda f: pd.read_csv(f"{CARPETA}/{f}.csv")
coco, npa, chem, chem2, pc = L("std_coconut"), L("std_npatlas"), L("std_chembl"), L("std_chembl_replica"), L("std_pubchem")
abio = L("lista_abiotica_nivelA_v1")

# ---------- piezas reutilizadas (a partir de la ruta de ensamblaje minima) ----------
import re, io
import assembly_theory as at
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
def pathway(smi):
    m = Chem.MolFromSmiles(smi); k = Chem.RWMol(m); Chem.Kekulize(k, clearAromaticFlags=True)
    ma, _, _, P = at.index_search(Chem.MolToMolBlock(k, kekulize=False), max_pathways=1)
    dot = P[0]
    nodes = {int(a): set(map(int, b.split(", "))) for a, b in re.findall(r'^\s*(\d+) \[ label = "\{([\d, ]+)\}" \]', dot, re.M)}
    edges = [(int(a), int(b), set(map(int, c.split(", ")))) for a, b, c in re.findall(r'^\s*(\d+) -> (\d+) \[ label = "\{([\d, ]+)\}" \]', dot, re.M)]
    return m, ma, nodes, edges
def reutilizados(nodes, edges):
    out = {}
    for a, b, loc in edges:
        if len(nodes[a]) >= 2: out.setdefault(a, []).append(frozenset(loc))
    frs = [(len(nodes[a]), sorted(set(v), key=min)) for a, v in out.items() if len(set(v)) >= 2]
    return sorted(frs, key=lambda x: -x[0])
PAL = [(0.93, 0.45, 0.0), (0.12, 0.45, 0.85), (0.8, 0.15, 0.45), (0.2, 0.6, 0.2)]
def dibujar(smi, w=520, h=360, max_frag=3):
    m, ma, nodes, edges = pathway(smi); frs = reutilizados(nodes, edges)
    col, usados, leyenda = {}, set(), []
    for tam, copias in frs:
        libres = [c for c in copias if not (c & usados)]
        if len(libres) < 2: continue
        color = PAL[len(leyenda) % len(PAL)]
        for i, c in enumerate(libres):
            f = [0.0, 0.5, 0.7, 0.8][min(i, 3)]
            tono = tuple(x + (1 - x) * f for x in color)
            for b in c: col[b] = tono
            usados |= c
        leyenda.append((tam, len(libres), color))
        if len(leyenda) >= max_frag: break
    d = rdMolDraw2D.MolDraw2DCairo(w, h); o = d.drawOptions(); o.bondLineWidth = 2; o.highlightBondWidthMultiplier = 14
    atoms = {}
    for b, c in col.items():
        bd = m.GetBondWithIdx(b); atoms[bd.GetBeginAtomIdx()] = c; atoms[bd.GetEndAtomIdx()] = c
    d.DrawMolecule(m, highlightAtoms=list(atoms), highlightAtomColors=atoms, highlightBonds=list(col), highlightBondColors=col, highlightAtomRadii={a: .35 for a in atoms})
    d.FinishDrawing(); return Image.open(io.BytesIO(d.GetDrawingText())), ma, leyenda

# ======================= FIGURA 1 =======================
fig = plt.figure(figsize=(7.2, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.35], wspace=0.32)
axA = fig.add_subplot(gs[0]); axA.axis("off"); fig.text(0.1, 0.93, "a", fontsize=11, fontweight="bold")
pares_c = L("std_pares_coconut")
par = [("CC1=CCC[C@H]2[C@](C)(C=C3CC(=O)NC3=O)[C@@H](C)CC[C@]12C", "Natural (COCONUT)", C["nat"]),
       ("Cc1cccc(N2CCC3(CCC(CC(=O)O)CC3)CC2)c1", "Sintético (ChEMBL)", C["chembl"])]
def ma_de(smi):
    can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    f = pares_c[[Chem.MolToSmiles(Chem.MolFromSmiles(x)) == can for x in pares_c.smiles_std]]
    return int(f.MA_std_max.iloc[0]) if len(f) else None
m0 = Chem.MolFromSmiles(par[0][0])
axA.text(0.5, 1.0, "Fórmula idéntica $\\mathrm{" + re.sub(r"(\d+)", r"_{\1}", CalcMolFormula(m0)) + "}$ y " + str(m0.GetNumBonds()) + " enlaces", transform=axA.transAxes, ha="center", fontsize=7.5, color="0.25")
for k, (smi, etq, col) in enumerate(par):
    img, ma_k, ley = dibujar(smi, w=620, h=380)
    y0 = 0.53 - 0.47 * k
    ia = axA.inset_axes([0.0, y0, 1.0, 0.36]); ia.imshow(img); ia.axis("off")
    axA.text(0.5, y0 + 0.37, f"{etq}   MA = {ma_k}", transform=axA.transAxes, ha="center", fontsize=8, color=col, fontweight="bold")
    axA.text(0.5, y0 - 0.025, "piezas reutilizadas: " + ", ".join(f"{t} enlaces ×{n}" for t, n, _ in ley), transform=axA.transAxes, ha="center", fontsize=6.5, color="0.3")
axA.text(0.5, -0.13, "Mismo color = misma pieza, construida una vez y reutilizada (tono oscuro/claro = copias).\nEl producto natural queda cubierto casi por completo por piezas reutilizadas.", transform=axA.transAxes, ha="center", fontsize=7, style="italic", color="0.25")
ax = fig.add_subplot(gs[1]); fig.text(0.43, 0.93, "b", fontsize=11, fontweight="bold")
def banda(df, col, etq, ls="-"):
    d = df.dropna(subset=["MA_std_max"]); g = d.groupby("enlaces_std").MA_std_max
    q = pd.DataFrame({"n": g.size(), "q1": g.quantile(.25), "med": g.median(), "q3": g.quantile(.75)}); q = q[q.n >= 15]
    ax.fill_between(q.index, q.q1, q.q3, color=col, alpha=.15, lw=0); ax.plot(q.index, q.med, color=col, lw=1.8, ls=ls, label=etq)
banda(pd.concat([chem, chem2]), C["chembl"], "Sintéticos: ChEMBL")
banda(pc, C["pubchem"], "Sintéticos: PubChem (aleatoria)", ls="--")
banda(pd.concat([coco, npa]), C["nat"], "Productos naturales: COCONUT + NPAtlas")
conf = abio[abio.estado.str.startswith("confirmada")]; tent = abio[abio.estado.str.startswith("tentativa")]
ax.scatter(conf.enlaces, conf.MA_max, s=16, color=C["abio"], edgecolor="k", lw=.3, zorder=5, label="Abióticas confirmadas (Bennu)")
ax.scatter(tent.enlaces, tent.MA_max, s=16, facecolor="white", edgecolor=C["abio"], lw=.9, zorder=5, label="Abióticas tentativas (Bennu)")
pah = pd.DataFrame({"b": [19, 21, 24, 27, 30], "lo": [6, 6, 6, 7, 6], "hi": [6, 7, 7, 7, 8]})
ax.errorbar(pah.b, (pah.lo + pah.hi) / 2, yerr=(pah.hi - pah.lo) / 2, fmt="s", ms=3.5, mfc="white", mec=C["abio"], ecolor=C["abio"], lw=.8, zorder=5,
            label="Abióticas probables (HAP de Murchison)")
ax.axhline(15, ls=":", color="k", lw=.8); ax.text(0.6, 15.3, "umbral biótico propuesto\n(requiere alto número de copias)", fontsize=6.3, va="bottom", color="0.25")
ax.axvspan(12.5, 41, color=C["abio"], alpha=.04, lw=0)
ax.text(40.5, 1.2, "ninguna molécula abiótica con\nestructura confirmada en esta región", fontsize=6.3, color=C["abio"], ha="right")
ax.set_xlim(0, 41); ax.set_ylim(0, 25); ax.set_xlabel("Número de enlaces"); ax.set_ylabel("Índice de ensamblaje (MA)")
ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, handlelength=1.8, columnspacing=1.2)
guardar(fig, "Figura2_concepto_y_mapa"); plt.close(fig)

# ======================= FIGURA 2 =======================
from sklearn.metrics import roc_auc_score
MED = [("frac_estereo", "Fracción de C estereogénicos", "A"), ("frac_sp3", "Fsp3", "A"), ("mc1", "Ramificación (MC1)", "A"),
       ("simetria", "Asimetría", "A"), ("bertz", "Bertz", "B"), ("comp_smiles", "Compresión: SMILES", "B"),
       ("comp_enlaces", "Compresión: lista de enlaces", "B"), ("comp_adyacencia", "Compresión: adyacencia", "B"), ("MA_std_max", "Índice de ensamblaje (MA)", "B")]
MK = [m for m, _, _ in MED]; COMPRES = ["comp_smiles", "comp_enlaces", "comp_adyacencia"]
def _extra(df):
    if "frac_estereo" in df.columns and "mc1" in df.columns: return df
    fe, mc = [], []
    for s in df.smiles_std:
        m = Chem.MolFromSmiles(s); nc = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "C")
        fe.append(len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)) / max(1, nc))
        mc.append(sum(1 for a in m.GetAtoms() if a.GetDegree() != 2) / m.GetNumAtoms())
    return df.assign(frac_estereo=fe, mc1=mc)
coco, npa, chem, chem2, pc = [_extra(x) for x in (coco, npa, chem, chem2, pc)]
def auc_est(d):
    o = {m: [] for m in MK}; w = []
    for _, s in d.groupby("enlaces_std"):
        a, b = s.y.sum(), (1 - s.y).sum()
        if a < 20 or b < 20: continue
        w.append(min(a, b)); [o[m].append(roc_auc_score(s.y, s[m])) for m in MK]
    return {m: np.average(o[m], weights=w) for m in MK}
def comparar(g1, g0, rep=200):
    d = pd.concat([g1.assign(y=1), g0.assign(y=0)]).dropna(subset=MK)
    d = d[(d.enlaces_std >= 15) & (d.enlaces_std <= 35)].reset_index(drop=True)
    b = auc_est(d); rng = np.random.default_rng(7); gr = [s for _, s in d.groupby("enlaces_std")]; bs = {m: [] for m in MK}; v = []
    for _ in range(rep):
        r = auc_est(pd.concat([s.sample(len(s), replace=True, random_state=int(rng.integers(1e9))) for s in gr]))
        [bs[m].append(r[m]) for m in MK]; v.append(abs(r["MA_std_max"] - .5) - max(abs(r[c] - .5) for c in COMPRES))
    out = {m: (b[m], *np.percentile(bs[m], [2.5, 97.5])) for m in MK}
    out["ventaja"] = (abs(b["MA_std_max"] - .5) - max(abs(b[c] - .5) for c in COMPRES), *np.percentile(v, [2.5, 97.5])); return out
cache = f"{SAL}/auc_figura2.json"
if os.path.exists(cache): R = {k: {m: tuple(v) for m, v in d.items()} for k, d in json.load(open(cache)).items()}
else:
    R = {"COCONUT vs ChEMBL": comparar(coco, chem), "NPAtlas vs ChEMBL": comparar(npa, chem2),
         "COCONUT vs PubChem": comparar(coco, pc), "NPAtlas vs PubChem": comparar(npa, pc), "PubChem vs ChEMBL (control)": comparar(pc, chem)}
    json.dump({k: {m: list(v) for m, v in d.items()} for k, d in R.items()}, open(cache, "w"))
ESTILO = {"COCONUT vs ChEMBL": (C["chembl"], "o"), "NPAtlas vs ChEMBL": (C["chembl"], "D"),
          "COCONUT vs PubChem": (C["pubchem"], "o"), "NPAtlas vs PubChem": (C["pubchem"], "D")}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.1})
y = np.arange(len(MED))[::-1]
ax.axhspan(y[3] - .5, y[0] + .5, color="0.93", lw=0); ax.axvline(.5, color="k", lw=.7, ls=":")
for j, (k, (col, mk)) in enumerate(ESTILO.items()):
    off = (j - 1.5) * 0.16
    for i, (m, _, _) in enumerate(MED):
        v, lo, hi = R[k][m]
        ax.errorbar(v, y[i] + off, xerr=[[v - lo], [hi - v]], fmt=mk, ms=3.6, color=col, mfc=col if "COCONUT" in k else "white", mew=.9, elinewidth=.8, capsize=0,
                    label=k.replace(" vs ", " frente a ") if i == 0 else None)
ax.set_yticks(y); ax.set_yticklabels([n for _, n, _ in MED])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlim(0.08, 0.92); ax.set_xlabel("AUC estratificado por tamaño (natural frente a sintético)")
ax.text(0.9, y[0] + 0.62, "naturales con valores mayores  →", ha="right", fontsize=6.8, color="0.3")
ax.text(0.1, y[0] + 0.62, "←  naturales con valores menores", ha="left", fontsize=6.8, color="0.3")
ax.text(0.11, y[3] - 0.38, "estereoquímica · forma · ramificación", fontsize=6.5, color="0.35", style="italic")
ax.text(0.6, y[4] - 0.1, "contenido de información del grafo", fontsize=6.5, color="0.35", style="italic")
ax.legend(fontsize=6.3, loc="lower right", bbox_to_anchor=(1.0, 0.0), handletextpad=0.3); letra(ax, "a", x=-0.42)
ctrl = R["PubChem vs ChEMBL (control)"]; ax2.axvline(.5, color="k", lw=.7, ls=":")
for i, (m, _, _) in enumerate(MED):
    v, lo, hi = ctrl[m]; ax2.errorbar(v, y[i], xerr=[[v - lo], [hi - v]], fmt="s", ms=3.6, color="0.35" if m != "MA_std_max" else "k", capsize=0, elinewidth=.8)
ax2.set_yticks(y); ax2.set_yticklabels([]); ax2.set_xlim(0.3, 0.7); ax2.set_xlabel("AUC (PubChem frente a ChEMBL)")
ax2.set_title("Control negativo:\ndos fuentes sintéticas", fontsize=7.5); letra(ax2, "b", x=-0.08)
guardar(fig, "Figura3_MA_frente_a_complejidad"); plt.close(fig)

# ======================= FIGURA 3 =======================
def pares(nombre):
    p = L(nombre)
    def k(s):
        m = Chem.MolFromSmiles(s); m = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumAtoms()); return CalcMolFormula(m) + "|" + str(m.GetNumBonds())
    p["par"] = [k(s) for s in p.smiles]
    p["clave_std"] = [CalcMolFormula(Chem.MolFromSmiles(s)) + "|" + str(int(e)) if e == e else None for s, e in zip(p.smiles_std, p.enlaces_std)]
    n = p[p.origen == "natural"].drop_duplicates("par").set_index("par"); s_ = p[p.origen == "sintetico"].drop_duplicates("par").set_index("par")
    j = n.join(s_, lsuffix="_n", rsuffix="_s", how="inner").dropna(subset=["MA_std_max_n", "MA_std_max_s"])
    return j[j.clave_std_n == j.clave_std_s]
PC_, PN_ = pares("std_pares_coconut"), pares("std_pares_npatlas")
def frac(x):
    a, b = int((x < 0).sum()), int((x > 0).sum()); ci = stats.binomtest(a, a + b).proportion_ci(method="exact"); return a / (a + b), ci.low, ci.high, a + b
coco_full = [f for f in os.listdir(CARPETA) if "coconut_csv" in f.lower()]
fig = plt.figure(figsize=(10.0 if coco_full else 7.2, 3.0)); gs = fig.add_gridspec(1, 3 if coco_full else 2, width_ratios=[1, 1.15, 0.9][: 3 if coco_full else 2], wspace=0.75 if coco_full else 0.55)
ax = fig.add_subplot(gs[0]); bins = np.arange(-10.5, 11.5, 1)
for j_, col, etq, st in [(PC_, C["nat"], "COCONUT–ChEMBL", "stepfilled"), (PN_, "k", "NPAtlas–ChEMBL", "step")]:
    d = j_.MA_std_max_n - j_.MA_std_max_s
    ax.hist(d, bins=bins, density=True, histtype=st, color=col, alpha=.45 if st == "stepfilled" else 1, lw=1, label=f"{etq} (n = {len(d)})")
ax.axvline(0, color="k", lw=.7, ls=":"); ax.set_xlabel("ΔMA (natural − sintético)"); ax.set_ylabel("Fracción de pares")
ax.set_ylim(0, ax.get_ylim()[1] * 1.35); ax.legend(fontsize=6.2, loc="upper left"); letra(ax, "a", x=-0.3)
ax = fig.add_subplot(gs[1]); filas = []
for j_, nom in [(PC_, "COCONUT"), (PN_, "NPAtlas")]:
    dm = j_.MA_std_max_n - j_.MA_std_max_s
    filas += [(nom, "MA (Kekulé máx.)", frac(dm)), (nom, "MA (Kekulé mín.)", frac(j_.MA_std_min_n - j_.MA_std_min_s)),
              (nom, "MA, simetría igual", frac(dm[(j_.simetria_n - j_.simetria_s).abs() < .02])),
              (nom, "Compresión (adyacencia)", frac(j_.comp_adyacencia_n - j_.comp_adyacencia_s)),
              (nom, "Índice de simetría*", frac(j_.simetria_n - j_.simetria_s))]
etiquetas = ["MA (Kekulé máx.)", "MA (Kekulé mín.)", "MA, simetría igual", "Compresión (adyacencia)", "Índice de simetría*"]
yy = np.arange(len(etiquetas))[::-1]
for nom, col, off, mk in [("COCONUT", C["nat"], .13, "o"), ("NPAtlas", "k", -.13, "D")]:
    for i, e in enumerate(etiquetas):
        f, lo, hi, n = [r for r in filas if r[0] == nom and r[1] == e][0][2]
        ax.errorbar(f, yy[i] + off, xerr=[[f - lo], [hi - f]], fmt=mk, ms=3.4, color=col, mfc=col if nom == "COCONUT" else "white", capsize=0, elinewidth=.8,
                    label=nom if i == 0 else None)
ax.axvline(.5, color="k", lw=.7, ls=":"); ax.set_yticks(yy); ax.set_yticklabels(etiquetas, fontsize=7); ax.set_xlim(.2, .8)
ax.set_xlabel("Fracción de pares con\nMA natural menor"); ax.legend(fontsize=6.3, loc="center left"); letra(ax, "b", x=-0.78)
if coco_full:
    ax = fig.add_subplot(gs[2]); vias = pd.read_csv(f"{CARPETA}/{coco_full[0]}", usecols=["identifier", "np_classifier_pathway"])
    t = PC_.assign(identifier=PC_.id_n.str.split("_", n=2).str[2]).merge(vias, on="identifier", how="left")
    t["via"] = t.np_classifier_pathway.fillna("Unclassified").str.split("|").str[0].replace({"Shikimates and Phenylpropanoids": "Siquimatos", "Amino acids and Peptides": "Aminoácidos/péptidos", "Alkaloids": "Alcaloides", "Terpenoids": "Terpenos", "Polyketides": "Policétidos", "Fatty acids": "Ácidos grasos", "Carbohydrates": "Carbohidratos", "Unclassified": "Sin clasificar"})
    fil = [(v, *frac(s.MA_std_max_n - s.MA_std_max_s)) for v, s in t.groupby("via") if ((s.MA_std_max_n - s.MA_std_max_s) != 0).sum() >= 25]
    f = pd.DataFrame(fil, columns=["via", "p", "lo", "hi", "n"]).sort_values("n")
    glob = frac(PC_.MA_std_max_n - PC_.MA_std_max_s)[0]
    ax.axvspan(glob - .001, glob + .001, color=C["nat"], alpha=.6); ax.axvline(.5, color="k", lw=.7, ls=":")
    ax.errorbar(f.p, range(len(f)), xerr=[f.p - f.lo, f.hi - f.p], fmt="o", ms=3.4, color=C["nat"], capsize=0, elinewidth=.8)
    ax.set_yticks(range(len(f))); ax.set_yticklabels([f"{v} ({n})" for v, n in zip(f.via, f.n)], fontsize=6.5)
    ax.set_xlim(.3, .95); ax.set_xlabel("Fracción con MA natural menor\n(pares COCONUT, por ruta)"); letra(ax, "c", x=-0.75)
fig.text(0.37, -0.1, "*Valor menor = molécula más simétrica; los productos naturales son menos simétricos que sus pares sintéticos.", fontsize=6.3, color="0.35")
guardar(fig, "Figura4_pares_formula_identica"); plt.close(fig)

# ======================= FIGURA 4: espacio quimico y exceso de MA =======================
from rdkit.Chem import rdFingerprintGenerator
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
C.update({"nat2": "#66BB6A"})
G = {"COCONUT": coco, "NPAtlas": npa, "ChEMBL": pd.concat([chem, chem2]), "PubChem": pc}
todo = pd.concat([g.assign(fuente=k) for k, g in G.items()]).dropna(subset=["MA_std_max"])
todo = todo[(todo.enlaces_std >= 15) & (todo.enlaces_std <= 35)]
ref = todo.groupby("enlaces_std").MA_std_max.median()
todo["exceso"] = todo.MA_std_max - todo.enlaces_std.map(ref)
# ---------- mapa ----------
mues = pd.concat([g.sample(min(900, len(g)), random_state=3) for _, g in todo.groupby("fuente")]).reset_index(drop=True)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in mues.smiles_std], dtype=np.float32)
emb = TSNE(n_components=2, metric="jaccard", init="random", perplexity=40, random_state=3).fit_transform(X.astype(bool))
mues["x"], mues["y"] = emb[:, 0], emb[:, 1]
fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.35, wspace=0.12)
ax = fig.add_subplot(gs[0, 0]); col = {"COCONUT": C["nat"], "NPAtlas": C["nat2"], "ChEMBL": C["chembl"], "PubChem": C["pubchem"]}
for k in ["PubChem", "ChEMBL", "NPAtlas", "COCONUT"]:
    s = mues[mues.fuente == k]; ax.scatter(s.x, s.y, s=3, color=col[k], alpha=.55, lw=0, label=k, rasterized=True)
ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
ax.legend(markerscale=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, handletextpad=0.1, columnspacing=0.8); ax.set_title("Espacio químico (huellas de Morgan, t-SNE)", fontsize=8.5)
ax.text(-0.02, 1.04, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")
ax = fig.add_subplot(gs[0, 1]); o = mues.sample(frac=1, random_state=1)
sc = ax.scatter(o.x, o.y, s=3, c=o.exceso.clip(-5, 5), cmap="RdBu_r", vmin=-5, vmax=5, lw=0, rasterized=True)
ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
cb = fig.colorbar(sc, ax=ax, fraction=.04, pad=.01); cb.set_label("MA por encima (+) o por debajo (−)\nde la mediana de su tamaño", fontsize=7.5)
ax.set_title("Mismo mapa, coloreado por índice de ensamblaje", fontsize=8.5); ax.text(-0.02, 1.04, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")
# ---------- crestas ----------
ax = fig.add_subplot(gs[1, :]); xs = np.linspace(-9, 9, 400); orden = [("COCONUT", C["nat"]), ("NPAtlas", C["nat2"]), ("PubChem", C["pubchem"]), ("ChEMBL", C["chembl"])]
for i, (k, c) in enumerate(orden):
    v = todo[todo.fuente == k].exceso.values; dens = gaussian_kde(v, bw_method=.35)(xs); base = (len(orden) - 1 - i) * 0.55
    ax.fill_between(xs, base, base + dens / dens.max() * 0.5, color=c, alpha=.65, lw=0); ax.plot(xs, base + dens / dens.max() * 0.5, color=c, lw=1)
    med = np.median(v); ax.plot([med, med], [base, base + 0.5], color="k", lw=1)
    ax.text(-9.2, base + 0.12, f"{k}  (mediana {med:+.0f})".replace("-", "−"), ha="right", fontsize=7.5)
ax.axvline(0, color="k", lw=.6, ls=":"); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlim(-9, 9)
ax.set_xlabel("Índice de ensamblaje respecto de la mediana de las moléculas con el mismo número de enlaces")
ax.text(-8.8, 2.25, "← construidas con más reutilización", fontsize=7, color="0.35"); ax.text(8.8, 2.25, "construidas con menos reutilización →", fontsize=7, color="0.35", ha="right")
ax.text(-0.2, 1.02, "c", transform=ax.transAxes, fontsize=11, fontweight="bold")
guardar(fig, "Figura5_espacio_quimico"); plt.close(fig)
