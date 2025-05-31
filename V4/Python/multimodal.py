import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Cargar el dataset original
df = pd.read_csv("dataset_correlacion.csv")

# Separar características y etiqueta
X = df.drop(columns=["emocion_etiqueta"])
y = df["emocion_etiqueta"]

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducir dimensionalidad con PCA
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Balancear con SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Diccionario para almacenar resultados
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', gamma='scale'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f"\n===== Modelo: {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))






import joblib

# Guardar modelos
joblib.dump(models["KNN"], "modelo_knn.joblib")
joblib.dump(models["SVM"], "modelo_svm.joblib")
joblib.dump(models["RandomForest"], "modelo_rf.joblib")

# Guardar PCA y scaler para predicciones nuevas
joblib.dump(pca, "pca_transform.joblib")
joblib.dump(scaler, "scaler_transform.joblib")







import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import joblib

# =============================
# 1. Cargar modelos y transformaciones
# =============================
knn = joblib.load("modelo_knn.joblib")
svm = joblib.load("modelo_svm.joblib")
rf  = joblib.load("modelo_rf.joblib")
pca = joblib.load("pca_transform.joblib")
scaler = joblib.load("scaler_transform.joblib")

# =============================
# 2. Cargar modelo VGG16 para extracción de features
# =============================
base_model = VGG16(weights='imagenet', include_top=True)
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def extract_image_features(image_path):
    img = Image.open(image_path).resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg_model.predict(x)
    return features.flatten()

# =============================
# 3. Ingreso de datos nuevos
# =============================
# Ruta de imagen
ruta_imagen = input("Ingrese la ruta de la imagen: ")
# Ingreso de sensores
pulso = float(input("Ingrese pulso cardiaco: "))
presion = float(input("Ingrese presión sistólica: "))
sudoracion = float(input("Ingrese sudoración (ml/h): "))

# =============================
# 4. Concatenar y procesar vector de entrada
# =============================
imagen_vec = extract_image_features(ruta_imagen)
entrada = np.array([[pulso, presion, sudoracion] + imagen_vec.tolist()])
entrada_escalada = scaler.transform(entrada)
entrada_pca = pca.transform(entrada_escalada)

# =============================
# 5. Predicción con los 3 modelos
# =============================
print("\n=== Predicción de emoción ===")
print("KNN:", knn.predict(entrada_pca)[0])
print("SVM:", svm.predict(entrada_pca)[0])
print("Random Forest:", rf.predict(entrada_pca)[0])







import joblib


metrics = {
    "KNN": classification_report(y_test, knn.predict(X_test), output_dict=True),
    "SVM": classification_report(y_test, svm.predict(X_test), output_dict=True),
    "RandomForest": classification_report(y_test, rf.predict(X_test), output_dict=True)
}

joblib.dump(metrics, "metrics_multimodal.joblib")
