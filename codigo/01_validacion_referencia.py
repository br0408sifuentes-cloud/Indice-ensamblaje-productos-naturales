# 01. Validación: reproduce los valores de MA publicados con assembly-theory (checks, coconut_55, gdb13_1201).
import glob
base = "assembly-theory/data"
for conjunto in ["checks", "coconut_55", "gdb13_1201"]:
    ref = pd.read_csv(f"{base}/{conjunto}/ma-index.csv")
    filas = [dict(id=f"{conjunto}_{r.file_name}", origen=conjunto,
                  smiles=Chem.MolToSmiles(Chem.MolFromMolFile(f"{base}/{conjunto}/{r.file_name}"))) for r in ref.itertuples()]
    res = procesar(pd.DataFrame(filas), f"{CARPETA}/validacion_{conjunto}.csv", guardar_cada=500)
    m = res.merge(ref.assign(id=lambda x: conjunto + "_" + x.file_name), on="id")
    print(conjunto, ": coinciden con MA_max", (m.assembly_idx == m.MA_max).sum(), "de", len(m))
