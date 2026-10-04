# Índice de ensamblaje molecular en productos naturales y compuestos sintéticos

Código y datos del artículo:

**Los productos naturales requieren menos pasos de ensamblaje que los compuestos sintéticos de fórmula idéntica: qué mide el índice de ensamblaje molecular**
Braulio Edinson Sifuentes Cuadros (Universidad Privada del Norte), 2026.

## Contenido

| Carpeta | Contenido |
|---|---|
| `codigo/` | Scripts en el orden en que se ejecutaron (00 a 14) y funciones comunes (`herramientas.py`) |
| `datos/` | Muestras analizadas con su índice de ensamblaje (MA) y todos los descriptores |
| `figuras/` | Figuras del artículo a 300 dpi |

## Cómo reproducir los resultados

Los cálculos se hicieron en Google Colab (Python 3.13.15) con assembly-theory 0.7.0, RDKit 2026.03.6, scikit-learn 1.6.1, statsmodels 0.15.0, SciPy 1.16.3, pandas 2.2.3, NumPy 2.1.3 y Matplotlib 3.10.0.

1. Ejecutar `codigo/00_configuracion.py` y luego `codigo/herramientas.py` en cada sesión.
2. Descargar en la carpeta de trabajo las bases públicas: COCONUT 2.0 (versión de septiembre de 2026), NPAtlas (versión de septiembre de 2024) y ChEMBL (archivo `chemreps`). PubChem se consulta automáticamente.
3. Ejecutar los scripts 01 a 14 en orden. Los análisis estadísticos (06, 08, 10 y 11) pueden ejecutarse directamente con los archivos de `datos/`.

Todas las muestras aleatorias usan semillas fijas (42, 7, 11 y 3).

## Nota técnica

En assembly-theory 0.7.0, una búsqueda interrumpida por el límite de tiempo sigue ejecutándose y altera los cálculos siguientes (ver el reporte #148 del repositorio del programa). Por eso `herramientas.py` calcula cada molécula en un proceso aislado.

## Licencias

- **Código:** licencia MIT (ver `LICENSE`).
- **Datos:** derivados de COCONUT (CC0), NPAtlas (CC BY 4.0), ChEMBL (CC BY-SA 3.0) y PubChem (dominio público). Por la condición de ChEMBL, los archivos de `datos/` se distribuyen bajo CC BY-SA 4.0. Citar las bases originales al reutilizarlos.

## Uso de inteligencia artificial

El diseño de los análisis y la escritura del código contaron con apoyo de los modelos Claude Opus 5.5 y Claude Fable 5.1 (Anthropic). Todas las decisiones y verificaciones fueron realizadas por el autor.
