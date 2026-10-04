# 05. Agrega organismo documentado, ruta biosintética (NPClassifier) y puntuación NP desde el archivo completo de COCONUT.
ruta_coco = f"{CARPETA}/{[f for f in os.listdir(CARPETA) if 'coconut_csv' in f.lower()][0]}"
columnas = ["identifier", "organisms", "dois", "collections", "annotation_level",
            "np_likeness", "np_classifier_pathway", "np_classifier_superclass", "contains_sugar"]
meta = pd.read_csv(ruta_coco, usecols=columnas)
muestra = pd.read_csv(f"{CARPETA}/coconut_muestra5000_v2.csv").merge(meta, left_on="id", right_on="identifier", how="left")
muestra["con_organismo"] = muestra.organisms.notna() & (muestra.organisms.astype(str).str.strip() != "")
print("Con organismo documentado:", muestra.con_organismo.sum(), "de", len(muestra))
muestra.to_csv(f"{CARPETA}/coconut_muestra5000_v3_procedencia.csv", index=False)
