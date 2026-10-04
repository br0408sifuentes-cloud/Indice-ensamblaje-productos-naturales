# Semana 5: respuestas a la revision (objeciones 1, 6 y 11).
#  (1) Estandarizacion unica para todas las bases y recalculo del MA donde el grafo cambie.
#  (6) Medidas de compresion sobre el GRAFO y un indice de SIMETRIA.
#  (11) Prueba estratificada estandar (van Elteren).
# Requiere los pasos 6, C y 14. Retoma solo si Colab se desconecta.
# PREDICCIONES REGISTRADAS ANTES DE EJECUTAR:
#  (P1) Tras estandarizar, la ventaja del MA sobre la mejor medida de compresion seguira siendo > 0 (IC 95% sin tocar 0).
#  (P2) En los pares que conservan formula identica tras estandarizar, el natural tendra menor MA en > 55%.
#  (P3) Exploratoria: la simetria explicara parte, pero no toda, la diferencia de MA.
import zlib, warnings
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
warnings.filterwarnings("ignore")
for _v in ["A", "B", "C", "D"]: globals().pop(_v, None)   # evita choque con la funcion C() de las formulas estadisticas

_limpiar = rdMolStandardize.CleanupParameters()
_taut = rdMolStandardize.TautomerEnumerator(); _taut.SetMaxTautomers(200); _taut.SetMaxTransforms(200)
_neutro = rdMolStandardize.Uncharger()
def estandarizar(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        m = rdMolStandardize.Cleanup(m, _limpiar)
        m = rdMolStandardize.FragmentParent(m)
        m = _neutro.uncharge(m)
        m = _taut.Canonicalize(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None
def plano(smi):
    m = Chem.MolFromSmiles(smi); return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None
_codigo = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
def medidas_grafo(smi):
    m = Chem.MolFromSmiles(smi); n = m.GetNumAtoms()
    orden = list(Chem.CanonicalRankAtoms(m, breakTies=True)); pos = {i: orden[i] for i in range(n)}
    A = np.zeros((n, n), dtype=np.uint8)
    for a in m.GetAtoms(): A[pos[a.GetIdx()], pos[a.GetIdx()]] = min(a.GetAtomicNum(), 255)
    ejes = []
    for b in m.GetBonds():
        i, j, t = pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()], _codigo.get(b.GetBondType(), 5)
        A[i, j] = A[j, i] = t; ejes.append(f"{min(i,j)}-{max(i,j)}-{t}")
    clases = len(set(Chem.CanonicalRankAtoms(m, breakTies=False)))
    return dict(comp_adyacencia=len(zlib.compress(A.tobytes(), 9)),
                comp_enlaces=len(zlib.compress(";".join(sorted(ejes)).encode(), 9)),
                comp_smiles=len(zlib.compress(plano(smi).encode(), 9)),
                simetria=round(clases / n, 4))

def preparar(df, etiqueta, archivo):
    """Estandariza, recalcula MA solo donde el grafo cambia, y agrega medidas de grafo. Guarda en CARPETA/archivo."""
    ruta = f"{CARPETA}/{archivo}"
    if os.path.exists(ruta):
        print("Ya existe (se reutiliza):", archivo); return pd.read_csv(ruta)
    d = df.copy(); d["smiles_std"] = [estandarizar(s) for s in d.smiles]; d = d.dropna(subset=["smiles_std"])
    d["cambio"] = [plano(a) != plano(b) for a, b in zip(d.smiles, d.smiles_std)]
    print(f"{etiqueta}: {len(d)} moleculas; el grafo cambia en {d.cambio.sum()} ({d.cambio.mean():.1%})")
    d["MA_std_max"], d["MA_std_min"], d["enlaces_std"] = d.MA_max, d.MA_min, d.enlaces
    for i in d.index[d.cambio]:
        r = medir(d.at[i, "smiles_std"], 10)
        d.loc[i, ["MA_std_max", "MA_std_min", "enlaces_std"]] = [r["MA_max"], r["MA_min"], r["enlaces"]] if r else [np.nan] * 3
    g = pd.DataFrame([medidas_grafo(s) for s in d.smiles_std], index=d.index)
    d = pd.concat([d.drop(columns=[c for c in g.columns if c in d.columns]), g], axis=1)
    d.to_csv(ruta, index=False); return d

# ---------- datos ----------
coco = pd.read_csv(f"{CARPETA}/coconut_muestra5000_v3_procedencia.csv"); coco = coco[coco.con_organismo == True]
chem = pd.read_csv(f"{CARPETA}/chembl_muestra5000.csv")
std_coco = preparar(coco, "COCONUT (organismo)", "std_coconut.csv"); std_chem = preparar(chem, "ChEMBL", "std_chembl.csv")
conjuntos = [("COCONUT vs ChEMBL", std_coco, std_chem)]
if os.path.exists(f"{CARPETA}/replica_muestras_MA.csv"):
    rep = pd.read_csv(f"{CARPETA}/replica_muestras_MA.csv")
    std_npa = preparar(rep[rep.origen == "natural_npatlas"], "NPAtlas", "std_npatlas.csv")
    std_chem_rep = preparar(rep[rep.origen == "sintetico_chembl"], "ChEMBL (replica)", "std_chembl_replica.csv")
    conjuntos.append(("NPAtlas vs ChEMBL", std_npa, std_chem_rep))

MED = ["MA_std_max", "bertz", "comp_smiles", "comp_adyacencia", "comp_enlaces", "simetria", "entropia"]
def auc_est(d):
    out = {m: [] for m in MED}; w = []
    for _, s in d.groupby("enlaces_std"):
        a, b = s.y.sum(), (1 - s.y).sum()
        if a < 20 or b < 20: continue
        w.append(min(a, b)); [out[m].append(roc_auc_score(s.y, s[m])) for m in MED]
    return {m: np.average(out[m], weights=w) for m in MED}
def van_elteren(d, med):
    T = E = V = 0.0
    for _, s in d.groupby("enlaces_std"):
        n1, n0 = int(s.y.sum()), int((1 - s.y).sum()); n = n1 + n0
        if n1 == 0 or n0 == 0: continue
        r = stats.rankdata(s[med]); w = 1 / (n + 1)
        T += w * r[s.y.values == 1].sum(); E += w * n1 * (n + 1) / 2
        t = np.unique(s[med], return_counts=True)[1]
        V += w ** 2 * n1 * n0 / 12 * ((n + 1) - (t ** 3 - t).sum() / (n * (n - 1)))
    z = (T - E) / np.sqrt(V); return z, 2 * stats.norm.sf(abs(z))

filas = []
for nombre, nat, sin in conjuntos:
    d = pd.concat([nat.assign(y=1), sin.assign(y=0)]).dropna(subset=["MA_std_max"])
    d = d[(d.enlaces_std >= 15) & (d.enlaces_std <= 35)].reset_index(drop=True)
    base = auc_est(d); rng = np.random.default_rng(3); grupos = [s for _, s in d.groupby("enlaces_std")]
    bs = {m: [] for m in MED}; ventaja = []
    compres = ["comp_smiles", "comp_adyacencia", "comp_enlaces"]
    for _ in range(300):
        r = auc_est(pd.concat([s.sample(len(s), replace=True, random_state=int(rng.integers(1e9))) for s in grupos]))
        [bs[m].append(r[m]) for m in MED]
        ventaja.append(abs(r["MA_std_max"] - .5) - max(abs(r[c] - .5) for c in compres))
    print(f"\n==== {nombre} (estandarizado; 15 a 35 enlaces) | naturales {int(d.y.sum())} | sinteticos {int((1-d.y).sum())} ====")
    for m in MED:
        lo, hi = np.percentile(bs[m], [2.5, 97.5]); print(f"  {m:16s} AUC {base[m]:.3f} [{lo:.3f}, {hi:.3f}]  separacion {abs(base[m]-.5):.3f}")
        filas.append(dict(conjunto=nombre, medida=m, auc=base[m], lo=lo, hi=hi))
    v = abs(base["MA_std_max"] - .5) - max(abs(base[c] - .5) for c in compres); lo, hi = np.percentile(ventaja, [2.5, 97.5])
    print(f"  (P1) Ventaja del MA sobre la MEJOR compresion: {v:.3f} [{lo:.3f}, {hi:.3f}]")
    z, p = van_elteren(d, "MA_std_max"); print(f"  Prueba de van Elteren (MA, estratificada por enlaces): z = {z:.1f}, p = {p:.1e}")
    COMP = "frac_hetero + anillos + tipos_atomo + frac_aromatica + frac_sp3 + halogenos"
    f0 = f"y ~ C(enlaces_std) + {COMP} + bertz + entropia + comp_smiles + comp_adyacencia + comp_enlaces + simetria"
    m0 = smf.logit(f0, d).fit(disp=0, maxiter=300); m1 = smf.logit(f0 + " + MA_std_max", d).fit(disp=0, maxiter=300)
    lr = 2 * (m1.llf - m0.llf); print(f"  MA ademas de todo (incluidas compresiones de grafo y simetria): LR = {lr:.1f}, p = {stats.chi2.sf(lr,1):.1e}")
    print(f"  Simetria mediana natural / sintetico: {d[d.y==1].simetria.median():.3f} / {d[d.y==0].simetria.median():.3f}  (menor = mas simetrico)")
pd.DataFrame(filas).to_csv(f"{CARPETA}/resultados_auc_estandarizado.csv", index=False)

# ---------- pares ----------
def pares(archivo, salida):
    p = preparar(pd.read_csv(f"{CARPETA}/{archivo}"), f"Pares {archivo}", salida)
    def clave_orig(smi):
        m = Chem.MolFromSmiles(smi); m = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumAtoms())
        return CalcMolFormula(m) + "|" + str(m.GetNumBonds())
    p["par"] = [clave_orig(s) for s in p.smiles]   # emparejamiento por la formula original (no por el numero del id)
    p["clave"] = [CalcMolFormula(Chem.MolFromSmiles(s)) + "|" + str(int(e)) if e == e else None for s, e in zip(p.smiles_std, p.enlaces_std)]
    w = p.pivot_table(index="par", columns="origen", values=["MA_std_max", "simetria"], aggfunc="first").dropna()
    k = p.pivot_table(index="par", columns="origen", values="clave", aggfunc="first").dropna()
    w = w.loc[w.index.intersection(k.index[k.natural == k.sintetico])]
    dm = w.MA_std_max.natural - w.MA_std_max.sintetico; ds = w.simetria.natural - w.simetria.sintetico
    def linea(x, etq):
        a, b, e = (x < 0).sum(), (x > 0).sum(), (x == 0).sum()
        if a + b == 0: print(f"  {etq}: sin pares con diferencia"); return
        t = stats.binomtest(int(a), int(a + b)); ci = t.proportion_ci(method="exact")
        print(f"  {etq}: pares {len(x)} | natural menor {a} | mayor {b} | empates {e} | fraccion {a/(a+b):.3f} [{ci.low:.3f}, {ci.high:.3f}] | p {t.pvalue:.1e}")
    print(f"\n==== Pares {archivo} (tras estandarizar; solo pares que siguen con formula identica) ====")
    linea(dm, "(P2) MA")
    linea(ds, "Simetria (en vez de MA)")
    print(f"  Correlacion de Spearman entre diferencia de simetria y diferencia de MA: {stats.spearmanr(ds, dm).statistic:.3f}")
    linea(dm[ds.abs() < 0.02], "(P3) MA en pares con simetria casi igual")
pares("pares_formula_MA.csv", "std_pares_coconut.csv")
if os.path.exists(f"{CARPETA}/replica_pares_MA.csv"): pares("replica_pares_MA.csv", "std_pares_npatlas.csv")
print("\nListo.")
