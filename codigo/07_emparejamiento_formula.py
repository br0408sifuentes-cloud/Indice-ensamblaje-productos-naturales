# Semana 4: emparejamiento por formula molecular exacta (COCONUT con organismo vs ChEMBL).
# Requiere haber ejecutado antes los pasos 6, C y 14 (usa medir, procesar y formas_kekule).
# PREDICCION REGISTRADA ANTES DE EJECUTAR: en mas del 60% de los pares con MA distinto,
# el producto natural tendra MENOR MA que la molecula sintetica de la misma formula.
import random
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

N_PARES = 1500          # pares a calcular (unos 3.000 calculos de MA)
N_CHEMBL = 400000       # moleculas de ChEMBL a revisar para buscar formulas
ENLACES_MIN, ENLACES_MAX = 15, 35
random.seed(42)

def buscar(p):
    return [f for f in os.listdir(CARPETA) if all(x in f.lower() for x in p)]

def formula_y_enlaces(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None, None
    m = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumAtoms())
    return CalcMolFormula(m), m.GetNumBonds()

# 1) Productos naturales con organismo documentado
ruta_coco = f"{CARPETA}/{buscar(['coconut_csv'])[0]}"
coco = pd.read_csv(ruta_coco, usecols=["identifier", "canonical_smiles", "organisms", "standard_inchi_key"])
coco = coco[coco.organisms.notna() & (coco.organisms.astype(str).str.strip() != "")]
print("COCONUT con organismo:", len(coco))
fe = [formula_y_enlaces(s) for s in coco.canonical_smiles]
coco["formula"] = [f for f, _ in fe]; coco["enlaces"] = [e for _, e in fe]
coco = coco[(coco.enlaces >= ENLACES_MIN) & (coco.enlaces <= ENLACES_MAX)]
print("  ... entre", ENLACES_MIN, "y", ENLACES_MAX, "enlaces:", len(coco))

# 2) ChEMBL sin esqueletos de COCONUT
ruta_chembl = f"{CARPETA}/{buscar(['chembl', 'chemreps'])[0]}"
chembl = pd.read_csv(ruta_chembl, sep="\t", usecols=["chembl_id", "canonical_smiles", "standard_inchi_key"]).dropna()
esqueletos_coco = set(pd.read_csv(ruta_coco, usecols=["standard_inchi_key"]).standard_inchi_key.dropna().str[:14])
chembl = chembl[~chembl.standard_inchi_key.str[:14].isin(esqueletos_coco)]
chembl = chembl.sample(n=min(N_CHEMBL, len(chembl)), random_state=42)
fe = [formula_y_enlaces(s) for s in chembl.canonical_smiles]
chembl["formula"] = [f for f, _ in fe]; chembl["enlaces"] = [e for _, e in fe]
chembl = chembl[(chembl.enlaces >= ENLACES_MIN) & (chembl.enlaces <= ENLACES_MAX)]
print("ChEMBL revisadas en el rango:", len(chembl))

# 3) Pares: misma formula Y mismo numero de enlaces; un par por combinacion
clave = lambda df: df.formula + "|" + df.enlaces.astype(str)
coco["clave"] = clave(coco); chembl["clave"] = clave(chembl)
comunes = sorted(set(coco.clave) & set(chembl.clave))
print("Combinaciones formula+enlaces compartidas:", len(comunes))
random.shuffle(comunes); comunes = comunes[:N_PARES]
g_coco = coco.groupby("clave"); g_chembl = chembl.groupby("clave")
filas = []
for n, k in enumerate(comunes):
    a = g_coco.get_group(k).sample(1, random_state=42).iloc[0]
    b = g_chembl.get_group(k).sample(1, random_state=42).iloc[0]
    filas.append(dict(id=f"par{n:04d}_nat_{a.identifier}", origen="natural", smiles=a.canonical_smiles, par=k))
    filas.append(dict(id=f"par{n:04d}_sin_{b.chembl_id}", origen="sintetico", smiles=b.canonical_smiles, par=k))
tabla_pares = pd.DataFrame(filas)
tabla_pares.to_csv(f"{CARPETA}/pares_formula_lista.csv", index=False)
print("Pares armados:", len(tabla_pares) // 2)

# 4) Calcular MA (retoma solo si se corta)
res = procesar(tabla_pares.drop(columns="par"), f"{CARPETA}/pares_formula_MA.csv")
res = res.merge(tabla_pares[["id", "par"]], on="id")

# 5) Prueba
from scipy import stats
ancho = res.pivot_table(index="par", columns="origen", values="MA_max").dropna()
dif = ancho.natural - ancho.sintetico
menor, mayor, empate = (dif < 0).sum(), (dif > 0).sum(), (dif == 0).sum()
print(f"\nPares completos: {len(ancho)}  | natural menor: {menor} | natural mayor: {mayor} | empates: {empate}")
print(f"Fraccion natural < sintetico (sin empates): {menor / (menor + mayor):.3f}   (prediccion registrada: > 0.60)")
print("Prueba de signo, p =", stats.binomtest(int(menor), int(menor + mayor), 0.5).pvalue)
print("Diferencia mediana de MA (natural - sintetico):", dif.median())
