# Analisis semana 3: productos naturales (COCONUT) vs sinteticos (ChEMBL), 15 a 35 enlaces.
# Uso en Colab: ejecuta los pasos 6, C y 14; luego:  exec(open(f"{CARPETA}/analisis_semana3.py").read())
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.formula.api as smf
from rdkit import Chem, RDConfig
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score"))
import npscorer
modelo_np = npscorer.readNPModel()

bio = pd.read_csv(f"{CARPETA}/coconut_muestra5000_v2.csv")
sin = pd.read_csv(f"{CARPETA}/chembl_muestra5000.csv")
bio["np_score"] = [npscorer.scoreMol(Chem.MolFromSmiles(s), modelo_np) for s in bio.smiles]
d = pd.concat([bio, sin]).dropna(subset=["MA_max"])
d = d[(d.enlaces >= 15) & (d.enlaces <= 35)].reset_index(drop=True)
d["y"] = (d.origen == "biologico_natural").astype(int)
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

def bootstrap(df, repeticiones=500, semilla=42):
    rng = np.random.default_rng(semilla); grupos = [s for _, s in df.groupby("enlaces")]
    guard = {m: [] for m in MED}; dif = []
    for _ in range(repeticiones):
        bs = pd.concat([s.sample(len(s), replace=True, random_state=int(rng.integers(1e9))) for s in grupos])
        r = auc_estratificado(bs)
        for m in MED: guard[m].append(r[m])
        dif.append(abs(r["MA_max"] - .5) - abs(r["compresion"] - .5))
    return guard, dif

def modelo_incremental(df, permutaciones=200):
    df = df.reset_index(drop=True)
    f0 = "y ~ C(enlaces) + " + " + ".join(COMP) + " + compresion + bertz + entropia"
    m0 = smf.logit(f0, df).fit(disp=0); m1 = smf.logit(f0 + " + MA_max", df).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf)
    rng = np.random.default_rng(1); nulos = []
    for _ in range(permutaciones):
        dp = df.copy()
        dp["MA_max"] = dp.groupby("enlaces").MA_max.transform(lambda s: rng.permutation(s.values))
        nulos.append(2 * (smf.logit(f0 + " + MA_max", dp).fit(disp=0).llf - m0.llf))
    p_perm = (np.sum(np.array(nulos) >= lr) + 1) / (permutaciones + 1)
    E = pd.get_dummies(df.enlaces.astype(str), prefix="k")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    def auc_cv(X):
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                              X, df.y, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(df.y, p)
    base = E.join(df[COMP + ["compresion", "bertz", "entropia"]])
    return dict(n_bio=int(df.y.sum()), n_sint=int((1 - df.y).sum()), LR=round(lr, 1),
                p_chi2=stats.chi2.sf(lr, 1), p_perm=round(p_perm, 4),
                coef_estandarizado=round(m1.params["MA_max"] * df.MA_max.std(), 2),
                auc_cv_base=round(auc_cv(base), 3), auc_cv_base_mas_MA=round(auc_cv(base.join(df[["MA_max"]])), 3),
                mediana_MA_bio=df[df.y == 1].MA_max.median(), mediana_MA_sint=df[df.y == 0].MA_max.median())

print("Moleculas por grupo:", d.groupby("origen").size().to_dict())
base = auc_estratificado(d); guard, dif = bootstrap(d)
print("\nAUC estratificado por numero exacto de enlaces (IC 95% bootstrap):")
for m in MED:
    lo, hi = np.percentile(guard[m], [2.5, 97.5])
    print(f"  {m:11s} {base[m]:.3f}  [{lo:.3f}, {hi:.3f}]   distancia a 0,5 = {abs(base[m]-.5):.3f}")
lo, hi = np.percentile(dif, [2.5, 97.5])
print(f"\nDistancia(MA) - distancia(compresion): {abs(base['MA_max']-.5)-abs(base['compresion']-.5):.3f}  [{lo:.3f}, {hi:.3f}]")
print("\nModelo incremental (MA ademas de tamano, composicion, compresion, Bertz y entropia):")
for etiqueta, sub in [("COCONUT completo", d),
                      ("Sensibilidad: COCONUT con np_score > 0", pd.concat([d[(d.y == 1) & (d.np_score > 0)], d[d.y == 0]])),
                      ("Sensibilidad: COCONUT con np_score <= 0", pd.concat([d[(d.y == 1) & (d.np_score <= 0)], d[d.y == 0]]))]:
    print(" ", etiqueta, modelo_incremental(sub))
