import joblib
import pandas as pd
from sklearn.metrics import classification_report
import joblib



def mostrar_metricas(nombre_archivo, nombre_modelo):
    try:
        metrics = joblib.load(nombre_archivo)
        print(f"\n📊 MÉTRICAS - {nombre_modelo.upper()}")
        for modelo, reporte in metrics.items():
            print(f"\n--- {modelo} ---")
            df = pd.DataFrame(reporte).transpose()
            print(df[['precision', 'recall', 'f1-score']].round(2))
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {nombre_archivo}")

# Cargar métricas de cada enfoque
mostrar_metricas("metrics_sensores.joblib", "Sensores")
mostrar_metricas("metrics_imagenes.joblib", "Imágenes")
mostrar_metricas("metrics_multimodal.joblib", "Multimodal")
