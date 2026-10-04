# Paso 3 del plan: REPLICACION con NPAtlas (productos naturales microbianos, curados, con organismo y referencia).
# Requiere haber ejecutado los pasos 6, C y 14. Retoma solo si Colab se desconecta.
# PREDICCIONES REGISTRADAS ANTES DE EJECUTAR:
#  (1) A igual tamano, el MA de NPAtlas sera menor que el de ChEMBL (AUC < 0.5) y superara a la compresion plana.
#  (2) En pares de formula identica NPAtlas-ChEMBL, el natural tendra menor MA en mas del 55% de los pares con diferencia.
import zlib, random, warnings
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
warnings.filterwarnings("ignore")
N_NAT, N_SIN, N_PARES, N_CHEMBL_PARES = 3000, 3000, 1000, 400000
random.seed(7)

def buscar(p): return [f for f in os.listdir(CARPETA) if all(x in f.lower() for x in p)]
def columna(df, pistas):
    for pista in pistas:
        for c in df.columns:
            if pista in c.lower(): return c
    return None
def base_mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    return max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumAtoms())
def info(s):
    m = base_mol(s)
    if m is None: return None, None, None
    return CalcMolFormula(m), m.GetNumBonds(), Chem.MolToInchiKey(m)[:14]
def comp_plana(s):
    return len(zlib.compress(Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False).encode(), 9))

# --- 1) NPAtlas ---
ruta_np = f"{CARPETA}/{buscar(['npatlas'])[0]}"
sep = "\t" if ruta_np.endswith((".tsv", ".txt")) else ","
npa = pd.read_csv(ruta_np, sep=sep, low_memory=False)
c_smi, c_id = columna(npa, ["compound_smiles", "smiles"]), columna(npa, ["npaid", "npa_id", "id"])
npa = npa[[c_id, c_smi]].dropna().rename(columns={c_id: "npaid", c_smi: "smiles"})
print("Moleculas en NPAtlas:", len(npa))
# excluir lo que ya usamos antes (para que la replicacion use moleculas nuevas)
usadas = set()
for f in ["coconut_muestra5000_v2.csv", "pares_formula_MA.csv"]:
    if os.path.exists(f"{CARPETA}/{f}"):
        for s in pd.read_csv(f"{CARPETA}/{f}").smiles:
            m = base_mol(s)
            if m is not None: usadas.add(Chem.MolToInchiKey(m)[:14])
x = [info(s) for s in npa.smiles]
npa["formula"], npa["enlaces"], npa["esq"] = zip(*x)
npa = npa[(npa.enlaces >= 15) & (npa.enlaces <= 35) & (~npa.esq.isin(usadas))]
print("NPAtlas nuevas entre 15 y 35 enlaces:", len(npa))

# --- 2) ChEMBL (muestra NUEVA, sin esqueletos de COCONUT ni de NPAtlas) ---
ruta_ch = f"{CARPETA}/{buscar(['chembl', 'chemreps'])[0]}"
ch = pd.read_csv(ruta_ch, sep="\t", usecols=["chembl_id", "canonical_smiles", "standard_inchi_key"]).dropna()
ruta_coco = f"{CARPETA}/{buscar(['coconut_csv'])[0]}"
esq_nat = set(pd.read_csv(ruta_coco, usecols=["standard_inchi_key"]).standard_inchi_key.dropna().str[:14]) | set(npa.esq) | usadas
ch = ch[~ch.standard_inchi_key.str[:14].isin(esq_nat)].sample(n=N_CHEMBL_PARES, random_state=7)
x = [info(s) for s in ch.canonical_smiles]
ch["formula"], ch["enlaces"], ch["esq"] = zip(*x)
ch = ch[(ch.enlaces >= 15) & (ch.enlaces <= 35)].rename(columns={"canonical_smiles": "smiles"})
print("ChEMBL candidatas entre 15 y 35 enlaces:", len(ch))

# --- 3) Muestras para la comparacion a igual tamano ---
m_nat = npa.sample(n=min(N_NAT, len(npa)), random_state=7)
m_sin = ch.sample(n=min(N_SIN, len(ch)), random_state=7)
tabla = pd.concat([pd.DataFrame({"id": "npa_" + m_nat.npaid.astype(str), "origen": "natural_npatlas", "smiles": m_nat.smiles}),
                   pd.DataFrame({"id": "chb_" + m_sin.chembl_id, "origen": "sintetico_chembl", "smiles": m_sin.smiles})])
res = procesar(tabla, f"{CARPETA}/replica_muestras_MA.csv")
agregar_composicion("replica_muestras_MA.csv")

# --- 4) Pares de formula identica ---
clave = lambda df: df.formula + "|" + df.enlaces.astype(str)
npa["clave"], ch["clave"] = clave(npa), clave(ch)
comunes = sorted(set(npa.clave) & set(ch.clave)); random.shuffle(comunes); comunes = comunes[:N_PARES]
gn, gc = npa.groupby("clave"), ch.groupby("clave"); filas = []
for n, k in enumerate(comunes):
    a = gn.get_group(k).sample(1, random_state=7).iloc[0]; b = gc.get_group(k).sample(1, random_state=7).iloc[0]
    filas += [dict(id=f"rpar{n:04d}_nat_{a.npaid}", origen="natural", smiles=a.smiles),
              dict(id=f"rpar{n:04d}_sin_{b.chembl_id}", origen="sintetico", smiles=b.smiles)]
pares = pd.DataFrame(filas); print("Pares armados:", len(pares) // 2)
res_p = procesar(pares, f"{CARPETA}/replica_pares_MA.csv")

# --- 5) Resultados ---
print("\n==== RESULTADO 1: comparacion a igual tamano (NPAtlas vs ChEMBL nuevo) ====")
d = pd.read_csv(f"{CARPETA}/replica_muestras_MA.csv").dropna(subset=["MA_max"])
d = d[(d.enlaces >= 15) & (d.enlaces <= 35)].reset_index(drop=True)
d["y"] = (d.origen == "natural_npatlas").astype(int); d["compresion"] = [comp_plana(s) for s in d.smiles]
MED = ["MA_max", "MA_min", "bertz", "compresion", "entropia"]
def auc_est(df):
    r = {m: [] for m in MED}; w = []
    for _, s in df.groupby("enlaces"):
        a, b = s.y.sum(), (1 - s.y).sum()
        if a < 20 or b < 20: continue
        w.append(min(a, b))
        for m in MED: r[m].append(roc_auc_score(s.y, s[m]))
    return {m: np.average(r[m], weights=w) for m in MED}
b = auc_est(d); rng = np.random.default_rng(7); grupos = [s for _, s in d.groupby("enlaces")]; dif = []; bst = {m: [] for m in MED}
for _ in range(300):
    r = auc_est(pd.concat([s.sample(len(s), replace=True, random_state=int(rng.integers(1e9))) for s in grupos]))
    for m in MED: bst[m].append(r[m])
    dif.append(abs(r["MA_max"] - .5) - abs(r["compresion"] - .5))
print("Moleculas:", d.groupby("origen").size().to_dict())
for m in MED: print(f"  {m:11s} {b[m]:.3f} {np.percentile(bst[m], [2.5, 97.5]).round(3)}")
print(f"  Ventaja MA sobre compresion: {abs(b['MA_max']-.5)-abs(b['compresion']-.5):.3f} {np.percentile(dif, [2.5, 97.5]).round(3)}")
COMP = "frac_hetero + anillos + tipos_atomo + frac_aromatica + frac_sp3 + halogenos"
f0 = f"y ~ C(enlaces) + {COMP} + compresion + bertz + entropia"
m0 = smf.logit(f0, d).fit(disp=0); m1 = smf.logit(f0 + " + MA_max", d).fit(disp=0); lr = 2 * (m1.llf - m0.llf)
print(f"  MA ademas de todo lo demas: LR = {lr:.1f}, p = {stats.chi2.sf(lr, 1):.1e}")
print(f"  Mediana MA natural / sintetico: {d[d.y==1].MA_max.median()} / {d[d.y==0].MA_max.median()}")

print("\n==== RESULTADO 2: pares de formula identica (NPAtlas vs ChEMBL) ====")
p = pd.read_csv(f"{CARPETA}/replica_pares_MA.csv"); p["par"] = p.id.str[:8]
w = p.pivot_table(index="par", columns="origen", values=["MA_max", "MA_min"]).dropna()
for med in ["MA_max", "MA_min"]:
    x = w[med].natural - w[med].sintetico; a, bb, e = (x < 0).sum(), (x > 0).sum(), (x == 0).sum()
    t = stats.binomtest(int(a), int(a + bb)); ci = t.proportion_ci(method="exact")
    print(f"  {med}: pares {len(x)} | natural menor {a} | mayor {bb} | empates {e} | fraccion {a/(a+bb):.3f} [{ci.low:.3f}, {ci.high:.3f}] | p {t.pvalue:.1e}")
