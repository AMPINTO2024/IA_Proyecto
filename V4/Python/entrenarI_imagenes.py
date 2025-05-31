import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Cargar vectores y nombres
data = np.load('vectores_imagenes_emociones.npz')
X = data['vectores']
nombres = data['nombres']

# Extraer etiquetas desde nombres (por ejemplo, 'felicidad/img1.jpg' → 'felicidad')
etiquetas = [nombre.split('/')[0] for nombre in nombres]

# Codificar etiquetas a números
le = LabelEncoder()
y = le.fit_transform(etiquetas)

# Escalar vectores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluar KNN
y_pred_knn = knn.predict(X_test)
print("Reporte de clasificación - KNN")
print(classification_report(y_test, y_pred_knn, target_names=le.classes_))
print("Exactitud:", accuracy_score(y_test, y_pred_knn))

# Otro método - SVM
svm = SVC(kernel='sigmoid', probability=True)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
print("\nReporte de clasificación - SVM")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))
print("Exactitud:", accuracy_score(y_test, y_pred_svm))

# Guardar modelos
joblib.dump(knn, "knn_model.joblib")
joblib.dump(svm, "svm_model.joblib")
joblib.dump(le, "label_encoder.joblib")
joblib.dump(scaler, "scaler_imagenes.joblib")

from sklearn.ensemble import RandomForestClassifier

# Entrenar modelo Random Forest
rf = RandomForestClassifier(n_estimators=20, random_state=42)
rf.fit(X_train, y_train)

# Evaluar modelo
y_pred_rf = rf.predict(X_test)

print("\nReporte de clasificación - Random Forest")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
print("Exactitud:", accuracy_score(y_test, y_pred_rf))

# Guardar modelo
joblib.dump(rf, "rf_model.joblib")
















