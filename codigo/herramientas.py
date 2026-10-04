# Herramientas comunes: cálculo del MA en procesos aislados, formas de Kekulé, medidas y descriptores.
# Ejecutar después de 00_configuracion.py en cada sesión.
import numpy as np
import zlib, math, time, os
import multiprocessing as mp
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import GraphDescriptors, rdFingerprintGenerator
import assembly_theory as at
RDLogger.DisableLog("rdApp.*")
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
_ctx = mp.get_context("fork")

def formas_kekule(mol, max_formas=8):
    bloques = set()
    try:
        for i, forma in enumerate(Chem.ResonanceMolSupplier(mol, Chem.KEKULE_ALL)):
            if i >= max_formas: break
            forma = Chem.RWMol(forma)
            for a in forma.GetAtoms(): a.SetIsAromatic(False)
            for b in forma.GetBonds(): b.SetIsAromatic(False)
            bloques.add(Chem.MolToMolBlock(forma, kekulize=False))
    except Exception:
        pass
    if not bloques:                                   # respaldo: forma por defecto de RDKit
        bloques.add(Chem.MolToMolBlock(mol))
    return list(bloques)

def _trabajador(bloques, cola):
    cola.put([at.index_search(b, parallel_str="none")[0] for b in bloques])

def ma_con_limite(bloques, limite_s):
    """Calcula el MA en un proceso aparte; si se pasa del limite, lo detiene de verdad."""
    cola = _ctx.Queue()
    p = _ctx.Process(target=_trabajador, args=(bloques, cola))
    p.start(); p.join(limite_s)
    if p.is_alive():
        p.kill(); p.join()
        return None
    return cola.get() if not cola.empty() else None

def entropia_entornos(mol):
    conteo = _gen.GetSparseCountFingerprint(mol).GetNonzeroElements()
    total = sum(conteo.values())
    return -sum((c/total)*math.log2(c/total) for c in conteo.values())

def medir(smiles, limite_s=10):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumAtoms())
    can = Chem.MolToSmiles(mol)
    valores = ma_con_limite(formas_kekule(mol), limite_s)
    return dict(smiles=can, atomos=mol.GetNumAtoms(), enlaces=mol.GetNumBonds(),
                MA_min=min(valores) if valores else None,
                MA_max=max(valores) if valores else None,
                n_kekule=len(valores) if valores else None,
                MA_incompleto=valores is None,
                bertz=round(GraphDescriptors.BertzCT(mol), 1),
                compresion=len(zlib.compress(can.encode(), 9)),
                entropia=round(entropia_entornos(mol), 3))

def procesar(tabla, archivo, guardar_cada=100, max_enlaces=40, limite_s=10):
    hechos = set(pd.read_csv(archivo)["id"]) if os.path.exists(archivo) else set()
    if hechos: print("Retomando: ya hay", len(hechos), "moleculas calculadas")
    nuevas, inicio = [], time.time()
    pendientes = tabla[~tabla["id"].isin(hechos)]
    for n, fila in enumerate(pendientes.itertuples(), 1):
        try:
            mol = Chem.MolFromSmiles(fila.smiles)
            if mol is not None and 2 <= mol.GetNumBonds() <= max_enlaces:
                r = medir(fila.smiles, limite_s)
                if r: nuevas.append(dict(id=fila.id, origen=fila.origen, **r))
        except Exception as e:
            print("Saltada", fila.id, "->", e)
        if n % guardar_cada == 0 or n == len(pendientes):
            if nuevas:
                pd.DataFrame(nuevas).to_csv(archivo, mode="a", index=False,
                                            header=not os.path.exists(archivo))
            nuevas = []
            print(f"{n}/{len(pendientes)} revisadas  ({time.time()-inicio:.0f} s)")
    return pd.read_csv(archivo)
from rdkit.Chem import rdMolDescriptors

def descriptores_composicion(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return {}
    pesados = mol.GetNumAtoms()
    hetero = sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H"))
    return dict(frac_hetero=round(hetero / pesados, 3),
                anillos=rdMolDescriptors.CalcNumRings(mol),
                tipos_atomo=len({a.GetSymbol() for a in mol.GetAtoms()}),
                frac_aromatica=round(sum(a.GetIsAromatic() for a in mol.GetAtoms()) / pesados, 3),
                frac_sp3=round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
                halogenos=sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")))

def _trabajador_cota(bloques, timeout_ms, cola):
    res = []
    for b in bloques:
        ma, _, estados, _ = at.index_search(b, timeout=timeout_ms, parallel_str="none")
        res.append((ma, estados is not None))
    cola.put(res)

def ma_con_cota(smiles, limite_s=120):
    mol = Chem.MolFromSmiles(smiles)
    mol = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumAtoms())
    bloques = formas_kekule(mol)
    cola = _ctx.Queue()
    por_forma_ms = max(1000, limite_s * 1000 // len(bloques))      # el limite total se reparte entre formas
    p = _ctx.Process(target=_trabajador_cota, args=(bloques, por_forma_ms, cola))
    p.start(); p.join(limite_s + 30)
    if p.is_alive():
        p.kill(); p.join(); return None
    return cola.get() if not cola.empty() else None

def repasar_incompletos(archivo_entrada, archivo_salida, limite_s=120):
    datos = pd.read_csv(archivo_entrada)
    if "MA_cota" not in datos.columns: datos["MA_cota"] = np.nan
    pendientes = datos.index[datos.MA_incompleto & datos.MA_cota.isna()]
    print("Moleculas por repasar:", len(pendientes))
    inicio = time.time()
    for n, i in enumerate(pendientes, 1):
        r = ma_con_cota(datos.at[i, "smiles"], limite_s)
        if r:
            valores = [ma for ma, _ in r]
            if all(ok for _, ok in r):
                datos.loc[i, ["MA_min", "MA_max", "MA_incompleto"]] = [min(valores), max(valores), False]
            else:
                datos.at[i, "MA_cota"] = max(valores)   # cota superior de MA_max
        else:
            datos.at[i, "MA_cota"] = -1
        datos.to_csv(archivo_salida, index=False)
        print(f"{n}/{len(pendientes)}  ({time.time()-inicio:.0f} s)")
    return datos

def agregar_composicion(nombre_archivo):
    ruta = f"{CARPETA}/{nombre_archivo}"
    d = pd.read_csv(ruta)
    comp = pd.DataFrame([descriptores_composicion(s) for s in d.smiles])
    d = pd.concat([d.drop(columns=[c for c in comp.columns if c in d.columns]), comp], axis=1)
    d.to_csv(ruta, index=False)
    print(nombre_archivo, "->", d.shape[0], "moleculas,", d.shape[1], "columnas")
