# 03. Muestra de ChEMBL sin esqueletos de COCONUT (5.000 moléculas, semilla 42) y cálculo del MA.
# Requiere el archivo chembl_<versión>_chemreps.txt.gz en CARPETA.
def buscar(palabras):
    return [f for f in os.listdir(CARPETA) if all(p in f.lower() for p in palabras)]

ruta_chembl = f"{CARPETA}/{buscar(['chembl', 'chemreps'])[0]}"
ruta_coco = f"{CARPETA}/{buscar(['coconut_csv'])[0]}"
print("ChEMBL:", ruta_chembl, "| COCONUT:", ruta_coco)

def columna(ruta, sep, pistas):
    cols = pd.read_csv(ruta, sep=sep, nrows=0).columns
    for pista in pistas:
        for c in cols:
            if pista in c.lower(): return c
    return None

c_smi = columna(ruta_chembl, "\t", ["canonical_smiles", "smiles"])
c_key = columna(ruta_chembl, "\t", ["inchi_key", "inchikey"])
c_id = columna(ruta_chembl, "\t", ["chembl_id"])
chembl = pd.read_csv(ruta_chembl, sep="\t", usecols=[c_id, c_smi, c_key]).dropna()
print("Moleculas en ChEMBL:", len(chembl))

k_coco = columna(ruta_coco, ",", ["inchi_key", "inchikey"])
s_coco = columna(ruta_coco, ",", ["canonical_smiles", "smiles"])
if k_coco:
    llaves_coco = set(pd.read_csv(ruta_coco, usecols=[k_coco])[k_coco].dropna().str[:14])
else:
    print("COCONUT no trae InChIKey; calculandolo (tarda varios minutos)...")
    smis = pd.read_csv(ruta_coco, usecols=[s_coco])[s_coco].dropna()
    llaves_coco = set()
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None: llaves_coco.add(Chem.MolToInchiKey(m)[:14])
print("Esqueletos unicos en COCONUT:", len(llaves_coco))

candidatas = chembl[~chembl[c_key].str[:14].isin(llaves_coco)]
print("ChEMBL sin coincidencias con COCONUT:", len(candidatas))
pre = candidatas.sample(n=min(30000, len(candidatas)), random_state=42)
pre["enlaces"] = [(m.GetNumBonds() if m is not None else -1)
                  for m in (Chem.MolFromSmiles(s) for s in pre[c_smi])]
pre = pre[(pre.enlaces >= 2) & (pre.enlaces <= 40)]
elegidas = pre.sample(n=min(5000, len(pre)), random_state=42)
tabla_sint = pd.DataFrame({"id": elegidas[c_id], "origen": "sintetico_chembl",
                           "smiles": elegidas[c_smi]})
print("Sinteticas elegidas:", len(tabla_sint))
res_sint = procesar(tabla_sint, f"{CARPETA}/chembl_muestra5000.csv")
print("Moleculas con tiempo agotado:", res_sint.MA_incompleto.sum())
agregar_composicion("chembl_muestra5000.csv")
