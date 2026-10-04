# Analisis semana 4. Requiere solo el Paso 6 (CARPETA definida).
# Parte A: repite la comparacion principal con compresion SIN simbolos de estereoquimica.
# Parte B: analiza los pares de formula identica (pares_formula_MA.csv).
import os, sys, zlib, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.formula.api as smf
from rdkit import Chem, RDLogger, RDConfig
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
sys.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score")); import npscorer
modelo_np = npscorer.readNPModel()

def compresion_plana(s):
    m = Chem.MolFromSmiles(s)
    return len(zlib.compress(Chem.MolToSmiles(m, isomericSmiles=False).encode(), 9))

# ---------------- Parte A ----------------
bio = pd.read_csv(f"{CARPETA}/coconut_muestra5000_v3_procedencia.csv")
sin = pd.read_csv(f"{CARPETA}/chembl_muestra5000.csv")
d = pd.concat([bio, sin]).dropna(subset=["MA_max"])
d = d[(d.enlaces >= 15) & (d.enlaces <= 35)].reset_index(drop=True)
d["y"] = (d.origen == "biologico_natural").astype(int)
d["compresion"] = [compresion_plana(s) for s in d.smiles]
COMP = ["frac_hetero", "anillos", "tipos_atomo", "frac_aromatica", "frac_sp3", "halogenos"]
MED = ["MA_max", "MA_min", "bertz", "compresion", "entropia"]

def auc_estratificado(df, minimo=20):
    res = {m: [] for m in MED}; pesos = []
    for _, s in df.groupby("enlaces"):
        a, b = s.y.sum(), (1 - s.y).sum()
        if a < minimo or b < minimo: continue
        pesos.append(min(a, b))
        for m in MED: res[m].append(roc_auc_score(s.y, s[m]))
    return {m: np.average(res[m], weights=pesos) for m in MED}

def bootstrap(df, rep=300, semilla=42):
    rng = np.random.default_rng(semilla); grupos = [s for _, s in df.groupby("enlaces")]
    g = {m: [] for m in MED}; dif = []
    for _ in range(rep):
        bs = pd.concat([s.sample(len(s), replace=True, random_state=int(rng.integers(1e9))) for s in grupos])
        r = auc_estratificado(bs)
        for m in MED: g[m].append(r[m])
        dif.append(abs(r["MA_max"] - .5) - abs(r["compresion"] - .5))
    return g, dif

def incremental(df, perm=100):
    df = df.reset_index(drop=True)
    f0 = "y ~ C(enlaces) + " + " + ".join(COMP) + " + compresion + bertz + entropia"
    m0 = smf.logit(f0, df).fit(disp=0); m1 = smf.logit(f0 + " + MA_max", df).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf); rng = np.random.default_rng(1); nulos = []
    for _ in range(perm):
        dp = df.copy(); dp["MA_max"] = dp.groupby("enlaces").MA_max.transform(lambda s: rng.permutation(s.values))
        nulos.append(2 * (smf.logit(f0 + " + MA_max", dp).fit(disp=0).llf - m0.llf))
    return dict(n_bio=int(df.y.sum()), n_sint=int((1 - df.y).sum()), LR=round(lr, 1), p_chi2=f"{stats.chi2.sf(lr, 1):.1e}",
                p_perm=round((np.sum(np.array(nulos) >= lr) + 1) / (perm + 1), 4))

print("PARTE A: compresion sin simbolos de estereoquimica")
for etiqueta, sub in [("COCONUT completo", d),
                      ("Solo con organismo documentado", pd.concat([d[(d.y == 1) & (d.con_organismo == True)], d[d.y == 0]]).reset_index(drop=True))]:
    b = auc_estratificado(sub); g, dif = bootstrap(sub)
    print(f"\n  {etiqueta}")
    for m in MED:
        lo, hi = np.percentile(g[m], [2.5, 97.5]); print(f"    {m:11s} {b[m]:.3f} [{lo:.3f}, {hi:.3f}]")
    lo, hi = np.percentile(dif, [2.5, 97.5])
    print(f"    Ventaja MA sobre compresion: {abs(b['MA_max']-.5)-abs(b['compresion']-.5):.3f} [{lo:.3f}, {hi:.3f}]")
    print("   ", incremental(sub))

# ---------------- Parte B ----------------
print("\nPARTE B: pares de formula identica")
p = pd.read_csv(f"{CARPETA}/pares_formula_MA.csv")
def clave(s):
    m = Chem.MolFromSmiles(s); m = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumAtoms())
    return CalcMolFormula(m) + "|" + str(m.GetNumBonds())
p["par"] = [clave(s) for s in p.smiles]
p["np"] = [npscorer.scoreMol(Chem.MolFromSmiles(s), modelo_np) for s in p.smiles]
p["comp_plana"] = [compresion_plana(s) for s in p.smiles]
w = p.pivot_table(index="par", columns="origen", values=["MA_max", "MA_min", "comp_plana", "np"]).dropna()
dm = w.MA_max.natural - w.MA_max.sintetico
def resumen(x, etq):
    a, b, e = (x < 0).sum(), (x > 0).sum(), (x == 0).sum()
    ci = stats.binomtest(int(a), int(a + b)).proportion_ci(method="exact")
    print(f"  {etq}: pares {len(x)} | natural menor {a} | mayor {b} | empates {e} | fraccion {a/(a+b):.3f} "
          f"[{ci.low:.3f}, {ci.high:.3f}] | p {stats.binomtest(int(a), int(a+b)).pvalue:.1e}")
resumen(dm, "MA_max")
resumen(w.MA_min.natural - w.MA_min.sintetico, "MA_min")
rng = np.random.default_rng(0); bs = [rng.choice(dm.values, len(dm)).mean() for _ in range(5000)]
print(f"  Diferencia media de MA (natural - sintetico): {dm.mean():.3f} [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
resumen(dm[w.np.sintetico < 0], "Solo pares con sintetico de aspecto sintetico (np < 0)")
dc = w.comp_plana.natural - w.comp_plana.sintetico
resumen(dc, "Compresion plana (en vez de MA)")
resumen(dm[dc == 0], "MA en pares con IGUAL compresion plana")
ruta_coco = [f for f in os.listdir(CARPETA) if "coconut_csv" in f.lower()]
if ruta_coco:
    vias = pd.read_csv(f"{CARPETA}/{ruta_coco[0]}", usecols=["identifier", "np_classifier_pathway"])
    nat = p[p.origen == "natural"].copy(); nat["identifier"] = nat.id.str.split("_", n=2).str[2]
    nat = nat.merge(vias, on="identifier", how="left")
    nat["via"] = nat.np_classifier_pathway.fillna("sin clasificar").str.split("|").str[0]
    nat = nat.set_index("par")
    print("\n  Por ruta biologica del producto natural:")
    for via, s in nat.groupby("via"):
        x = dm.reindex(s.index).dropna()
        if len(x) >= 30: resumen(x, via)
