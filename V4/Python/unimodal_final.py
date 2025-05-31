import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
import numpy as np

# Diccionario para convertir etiquetas a nombres de emociones
emociones_map = {
    0: 'Enojo',
    2: 'Miedo',
    3: 'Felicidad',
    4: 'Tristeza',
    5: 'Sorpresa'
}

# 1. Cargar el dataset
df = pd.read_csv('dataset_3sensores_2025.csv')    # Dataset solo con datos de 186 usuarios. 47%
                                                   #que a posterior son los que tienen imagen asociada
#df = pd.read_csv('dataset_con_clusters_y_emociones.csv') #Dataset con 600 datos de sensores. 96%

# 2. Seleccionar variables predictoras y etiqueta
X = df[['Pulso_Cardiaco', 'Presion_sistolica', 'Sudoracion_ml_h']]
y = df['emocion_etiqueta']

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Escalar variables (importante para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 6. Evaluar el modelo
y_pred_knn = knn.predict(X_test_scaled)
print("Reporte de clasificación KNN:")
print(classification_report(y_test, y_pred_knn))
print("Exactitud KNN:", accuracy_score(y_test, y_pred_knn))

# 7. Guardar el modelo y el escalador
dump(knn, 'modelo_knn.joblib')
dump(scaler, 'scaler.joblib')

# --- Cuando quieras usar el modelo guardado para predecir nuevos datos ---

# Cargar modelo y escalador
knn_cargado = load('modelo_knn.joblib')
scaler_cargado = load('scaler.joblib')



# Supongamos que recibes estos valores:
pulso_cardiaco = 100
presion_sistolica = 120
sudoracion_ml_h = 1.0

# Crear array numpy con estos valores (orden debe coincidir con el entrenamiento)
nuevo_dato = np.array([[pulso_cardiaco, presion_sistolica, sudoracion_ml_h]])

# Escalar el dato con el scaler cargado
nuevo_dato_scaled = scaler_cargado.transform(nuevo_dato)

# Predecir
prediccion_num = knn_cargado.predict(nuevo_dato_scaled)[0]  # obtener solo el primer valor

# Convertir a texto con el diccionario
prediccion_texto = emociones_map[prediccion_num]

print("La emoción predicha para el paciente es:", prediccion_texto)
















import pandas as pd
import numpy as np
from collections import Counter

# Modelos y utilidades
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from joblib import dump

# Mapeo opcional para mostrar texto
emociones_map = {
    0: 'Enojo',
    2: 'Miedo',
    3: 'Felicidad',
    4: 'Tristeza',
    5: 'Sorpresa'
}

# 1. Cargar dataset
df = pd.read_csv('dataset_3sensores_2025.csv')    # Dataset solo con datos de 186 usuarios. 47%
                                                   #que a posterior son los que tienen imagen asociada
#df = pd.read_csv('dataset_con_clusters_y_emociones.csv') #Dataset con 600 datos de sensores. 96%


# 2. Variables predictoras y etiqueta
X = df[['Pulso_Cardiaco', 'Presion_sistolica', 'Sudoracion_ml_h']]
y = df['emocion_etiqueta']

# Mostrar conteo original
print("Distribución original de etiquetas:", Counter(y))

# 3. Dividir con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Balancear con SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
print("Distribución después de SMOTE:", Counter(y_train_bal))

# 6. GridSearchCV para KNN
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_bal, y_train_bal)

best_knn = grid.best_estimator_
print(f"\nMejor KNN (k={grid.best_params_['n_neighbors']}):")
y_pred_knn = best_knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred_knn, target_names=[emociones_map.get(e, str(e)) for e in np.unique(y)]))
print("Exactitud KNN:", accuracy_score(y_test, y_pred_knn))

# 7. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bal, y_train_bal)
y_pred_rf = rf.predict(X_test_scaled)
print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf, target_names=[emociones_map.get(e, str(e)) for e in np.unique(y)]))
print("Exactitud RF:", accuracy_score(y_test, y_pred_rf))

# 8. SVM
svm = SVC(kernel='sigmoid', C=0.7, gamma='scale')
svm.fit(X_train_bal, y_train_bal)
y_pred_svm = svm.predict(X_test_scaled)
print("\nSVM:")
print(classification_report(y_test, y_pred_svm, target_names=[emociones_map.get(e, str(e)) for e in np.unique(y)]))
print("Exactitud SVM:", accuracy_score(y_test, y_pred_svm))

# 9. Guardar mejores modelos y scaler
dump(best_knn, 'modelo_knn_optimo.joblib')
dump(rf, 'modelo_random_forest.joblib')
dump(svm, 'modelo_svm.joblib')
dump(scaler, 'scaler.joblib')









from sklearn.metrics import classification_report
import joblib

# Evaluar y guardar las métricas
metrics = {
    "KNN": classification_report(y_test, best_knn.predict(X_test), output_dict=True),
    "SVM": classification_report(y_test, svm.predict(X_test), output_dict=True),
    "RandomForest": classification_report(y_test, rf.predict(X_test), output_dict=True)
}

# Guardar métricas en archivo
joblib.dump(metrics, "metrics_sensores.joblib")
print("✅ Métricas de sensores guardadas en metrics_sensores.joblib")






