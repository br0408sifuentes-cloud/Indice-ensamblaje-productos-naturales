# 00. Configuración del entorno (Google Colab). Ejecutar primero en cada sesión.
!pip install -q assembly-theory==0.7.0 rdkit pandas matplotlib scikit-learn statsmodels scipy
!git clone -q --depth 1 https://github.com/DaymudeLab/assembly-theory.git
from google.colab import drive
drive.mount('/content/drive')
import os
CARPETA = "/content/drive/MyDrive/Tabla1"   # carpeta con los datos descargados y los resultados
os.makedirs(CARPETA, exist_ok=True)
print("Carpeta de trabajo:", CARPETA)
