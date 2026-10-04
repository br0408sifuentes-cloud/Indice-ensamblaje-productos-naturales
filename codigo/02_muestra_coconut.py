# 02. Muestra aleatoria de COCONUT 2.0 (5.000 entradas, semilla 42) y cálculo del MA.
ruta = f"{CARPETA}/{[f for f in os.listdir(CARPETA) if 'coconut' in f.lower() and f.endswith(('.csv', '.zip'))][0]}"
coco = pd.read_csv(ruta, usecols=["identifier", "canonical_smiles"]).dropna()
muestra = coco.sample(n=5000, random_state=42)
tabla = pd.DataFrame({"id": muestra.identifier, "origen": "biologico_natural", "smiles": muestra.canonical_smiles})
procesar(tabla, f"{CARPETA}/coconut_muestra5000.csv")
agregar_composicion("coconut_muestra5000.csv")
