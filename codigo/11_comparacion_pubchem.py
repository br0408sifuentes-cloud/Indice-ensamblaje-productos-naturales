# Bloque C (objecion 3): productos naturales frente a una muestra aleatoria de PubChem.
# Requiere los pasos 6, C y 14, y haber ejecutado antes control_estandarizacion.py (usa sus archivos std_*.csv).
# PREDICCIONES REGISTRADAS ANTES DE EJECUTAR:
#  (P4) A igual tamano, los productos naturales (COCONUT con organismo y NPAtlas) tendran menor MA que la muestra
#       de PubChem (AUC < 0.5), y la ventaja del MA sobre la mejor compresion sera > 0 (IC 95% sin tocar 0).
#  (P5) Exploratoria: PubChem quedara entre los productos naturales y ChEMBL en MA a igual tamano.
import time, zlib, random, io, urllib.request, urllib.error, warnings
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
from rdkit.Chem.MolStandardize import rdMolStandardize
warnings.filterwarnings("ignore")
for _v in ["A", "B", "C", "D"]: globals().pop(_v, None)
N_OBJETIVO, CID_MAX, LOTE = 4000, 160_000_000, 200
random.seed(11)

# ---------- utilidades (iguales a control_estandarizacion.py) ----------
_limpiar = rdMolStandardize.CleanupParameters()
_taut = rdMolStandardize.TautomerEnumerator(); _taut.SetMaxTautomers(200); _taut.SetMaxTransforms(200)
_neutro = rdMolStandardize.Uncharger()
def estandarizar(smi):
    try:
        m = Chem.MolFromSmiles(smi); m = rdMolStandardize.Cleanup(m, _limpiar); m = rdMolStandardize.FragmentParent(m)
        m = _neutro.uncharge(m); m = _taut.Canonicalize(m); return Chem.MolToSmiles(m)
    except Exception:
        return None
_codigo = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
def medidas_grafo(smi):
    m = Chem.MolFromSmiles(smi); n = m.GetNumAtoms()
    orden = list(Chem.CanonicalRankAtoms(m, breakTies=True)); pos = {i: orden[i] for i in range(n)}
    A_ = np.zeros((n, n), dtype=np.uint8); ejes = []
    for a in m.GetAtoms(): A_[pos[a.GetIdx()], pos[a.GetIdx()]] = min(a.GetAtomicNum(), 255)
    for b in m.GetBonds():
        i, j, t = pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()], _codigo.get(b.GetBondType(), 5)
        A_[i, j] = A_[j, i] = t; ejes.append(f"{min(i,j)}-{max(i,j)}-{t}")
    plano = Chem.MolToSmiles(m, isomericSmiles=False)
    return dict(comp_adyacencia=len(zlib.compress(A_.tobytes(), 9)), comp_enlaces=len(zlib.compress(";".join(sorted(ejes)).encode(), 9)),
                comp_smiles=len(zlib.compress(plano.encode(), 9)), simetria=round(len(set(Chem.CanonicalRankAtoms(m, breakTies=False))) / n, 4))

# ---------- 1) descargar una muestra aleatoria de PubChem ----------
URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/CSV"
def descargar(cids):
    for props in ["SMILES,InChIKey", "IsomericSMILES,InChIKey", "CanonicalSMILES,InChIKey"]:
        try:
            req = urllib.request.Request(URL.format(",".join(map(str, cids)), props), headers={"User-Agent": "Tabla1-proyecto"})
            with urllib.request.urlopen(req, timeout=60) as r:
                t = pd.read_csv(io.StringIO(r.read().decode()))
            col = [c for c in t.columns if "SMILES" in c][0]
            return t.rename(columns={col: "smiles"})[["CID", "smiles", "InChIKey"]]
        except urllib.error.HTTPError as e:
            if e.code in (400,): continue          # nombre de propiedad no aceptado: probar el siguiente
            if e.code == 404: return pd.DataFrame(columns=["CID", "smiles", "InChIKey"])
            time.sleep(5)
        except Exception:
            time.sleep(5)
    return pd.DataFrame(columns=["CID", "smiles", "InChIKey"])

ruta_pc = f"{CARPETA}/pubchem_muestra_cruda.csv"
ruta_coco = [f for f in os.listdir(CARPETA) if "coconut_csv" in f.lower()][0]
esq_nat = set(pd.read_csv(f"{CARPETA}/{ruta_coco}", usecols=["standard_inchi_key"]).standard_inchi_key.dropna().str[:14])
if os.path.exists(f"{CARPETA}/std_npatlas.csv"):
    for s in pd.read_csv(f"{CARPETA}/std_npatlas.csv").smiles_std:
        m = Chem.MolFromSmiles(s)
        if m is not None: esq_nat.add(Chem.MolToInchiKey(m)[:14])
crudo = pd.read_csv(ruta_pc) if os.path.exists(ruta_pc) else pd.DataFrame(columns=["CID", "smiles", "InChIKey", "enlaces"])
def utiles(df):
    return df[(df.enlaces >= 15) & (df.enlaces <= 35) & (~df.InChIKey.astype(str).str[:14].isin(esq_nat))]
intentos = 0
while len(utiles(crudo)) < N_OBJETIVO and intentos < 600:
    lote = descargar(random.sample(range(1, CID_MAX), LOTE)); intentos += 1
    if len(lote):
        lote["enlaces"] = [(Chem.MolFromSmiles(s).GetNumBonds() if Chem.MolFromSmiles(s) is not None and "." not in s else -1) for s in lote.smiles]
        crudo = pd.concat([crudo, lote], ignore_index=True).drop_duplicates("CID"); crudo.to_csv(ruta_pc, index=False)
    if intentos % 10 == 0: print(f"Lotes {intentos}: {len(crudo)} descargadas, {len(utiles(crudo))} utiles (15-35 enlaces, sin esqueleto natural)")
    time.sleep(0.4)                                  # respeta el limite de PubChem (menos de 5 consultas por segundo)
pc = utiles(crudo).sample(n=min(N_OBJETIVO, len(utiles(crudo))), random_state=11)
print("Muestra de PubChem para analizar:", len(pc))

# ---------- 2) estandarizar y calcular MA ----------
ruta_std = f"{CARPETA}/std_pubchem.csv"
if os.path.exists(ruta_std):
    std_pc = pd.read_csv(ruta_std); print("Ya existe (se reutiliza): std_pubchem.csv")
else:
    t = pd.DataFrame({"id": "pc_" + pc.CID.astype(str), "origen": "pubchem", "smiles": [estandarizar(s) for s in pc.smiles]}).dropna()
    res = procesar(t, f"{CARPETA}/pubchem_MA.csv"); agregar_composicion("pubchem_MA.csv")
    std_pc = pd.read_csv(f"{CARPETA}/pubchem_MA.csv")
    std_pc["smiles_std"] = std_pc.smiles; std_pc["MA_std_max"], std_pc["MA_std_min"], std_pc["enlaces_std"] = std_pc.MA_max, std_pc.MA_min, std_pc.enlaces
    g = pd.DataFrame([medidas_grafo(s) for s in std_pc.smiles_std], index=std_pc.index)
    std_pc = pd.concat([std_pc, g], axis=1); std_pc.to_csv(ruta_std, index=False)

# ---------- 3) comparaciones ----------
MED = ["MA_std_max", "bertz", "comp_smiles", "comp_adyacencia", "comp_enlaces", "simetria", "entropia"]
COMPRES = ["comp_smiles", "comp_adyacencia", "comp_enlaces"]
def auc_est(d):
    out = {m: [] for m in MED}; w = []
    for _, s in d.groupby("enlaces_std"):
        a, b = s.y.sum(), (1 - s.y).sum()
        if a < 20 or b < 20: continue
        w.append(min(a, b)); [out[m].append(roc_auc_score(s.y, s[m])) for m in MED]
    return {m: np.average(out[m], weights=w) for m in MED}
def comparar(nombre, grupo1, grupo0):
    d = pd.concat([grupo1.assign(y=1), grupo0.assign(y=0)]).dropna(subset=["MA_std_max"])
    d = d[(d.enlaces_std >= 15) & (d.enlaces_std <= 35)].reset_index(drop=True)
    base = auc_est(d); rng = np.random.default_rng(5); gr = [s for _, s in d.groupby("enlaces_std")]; bs = {m: [] for m in MED}; v = []
    for _ in range(300):
        r = auc_est(pd.concat([s.sample(len(s), replace=True, random_state=int(rng.integers(1e9))) for s in gr]))
        [bs[m].append(r[m]) for m in MED]; v.append(abs(r["MA_std_max"] - .5) - max(abs(r[c] - .5) for c in COMPRES))
    print(f"\n==== {nombre} | grupo 1: {int(d.y.sum())} | grupo 0: {int((1-d.y).sum())} ====")
    for m in MED:
        lo, hi = np.percentile(bs[m], [2.5, 97.5]); print(f"  {m:16s} AUC {base[m]:.3f} [{lo:.3f}, {hi:.3f}]")
    lo, hi = np.percentile(v, [2.5, 97.5])
    print(f"  Ventaja del MA sobre la MEJOR compresion: {abs(base['MA_std_max']-.5)-max(abs(base[c]-.5) for c in COMPRES):.3f} [{lo:.3f}, {hi:.3f}]")
    COMP = "frac_hetero + anillos + tipos_atomo + frac_aromatica + frac_sp3 + halogenos"
    f0 = f"y ~ C(enlaces_std) + {COMP} + bertz + entropia + comp_smiles + comp_adyacencia + comp_enlaces + simetria"
    m0 = smf.logit(f0, d).fit(disp=0, maxiter=300); m1 = smf.logit(f0 + " + MA_std_max", d).fit(disp=0, maxiter=300); lr = 2 * (m1.llf - m0.llf)
    print(f"  MA ademas de todo: LR = {lr:.1f}, p = {stats.chi2.sf(lr,1):.1e}")
    print(f"  Mediana MA grupo 1 / grupo 0: {d[d.y==1].MA_std_max.median()} / {d[d.y==0].MA_std_max.median()}")
std_coco = pd.read_csv(f"{CARPETA}/std_coconut.csv"); std_chem = pd.read_csv(f"{CARPETA}/std_chembl.csv")
comparar("(P4) COCONUT (organismo) vs PubChem", std_coco, std_pc)
if os.path.exists(f"{CARPETA}/std_npatlas.csv"):
    comparar("(P4) NPAtlas vs PubChem", pd.read_csv(f"{CARPETA}/std_npatlas.csv"), std_pc)
comparar("(P5) PubChem vs ChEMBL", std_pc, std_chem)
print("\nListo.")
