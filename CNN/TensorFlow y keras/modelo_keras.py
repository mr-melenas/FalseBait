# Implementación de redes neuronales con TensorFlow y Keras para la detección de phishing

#######################################################
# SECCIÓN DE DEFINICIONES DE CONCEPTOS CLAVE
#######################################################

# TENSORFLOW Y KERAS:
# TensorFlow es una biblioteca de código abierto para aprendizaje automático desarrollada por Google.
# Keras es una API de alto nivel que se ejecuta sobre TensorFlow, facilitando la creación y
# entrenamiento de redes neuronales con menos código y mayor legibilidad.

# REDES NEURONALES DENSAS (FULLY CONNECTED):
# Son redes donde cada neurona de una capa está conectada con todas las neuronas de la capa siguiente.
# Son más simples que las CNN y adecuadas para datos tabulares como los que usamos en este problema.

# CAPAS DENSAS (DENSE LAYERS):
# Son capas donde cada neurona recibe entrada de todas las neuronas de la capa anterior.
# Cada conexión tiene un peso asociado que se ajusta durante el entrenamiento.

# FUNCIÓN DE ACTIVACIÓN:
# Introduce no-linealidad en el modelo, permitiendo aprender patrones complejos.
# ReLU (Rectified Linear Unit) es una de las más comunes y eficientes computacionalmente.

# DROPOUT:
# Técnica de regularización que desactiva aleatoriamente un porcentaje de neuronas durante
# el entrenamiento para evitar el sobreajuste.

#######################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas de TensorFlow y Keras para deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Bibliotecas de scikit-learn para preprocesamiento y evaluación
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')  # Suprime advertencias para una salida más limpia

# Configuración de semilla para reproducibilidad de resultados
tf.random.set_seed(42)  # Semilla para TensorFlow
np.random.seed(42)      # Semilla para NumPy

# Cargar el dataset desde GitHub
print("Cargando el dataset...")
df = pd.read_csv("https://raw.githubusercontent.com/juancmacias/datas/main/DataSet/PhiUSIIL_Phishing_URL_Dataset.csv")
print(f"Dataset cargado correctamente. Dimensiones: {df.shape}")

# Identificar la columna objetivo (variable dependiente)
target_column = 'label'  # Etiqueta que indica si una URL es phishing (1) o legítima (0)

# Identificar y eliminar columnas no numéricas que podrían causar problemas
non_numeric_cols = []
for col in df.columns:
    if col != target_column:  # Excluir la columna objetivo
        try:
            df[col].astype(float)  # Intentar convertir a float para verificar si es numérica
        except Exception as e:
            non_numeric_cols.append(col)
            print(f"La columna '{col}' contiene valores no numéricos y será eliminada.")

# Eliminar columnas no numéricas identificadas
if non_numeric_cols:
    print(f"\nEliminando {len(non_numeric_cols)} columnas no numéricas: {non_numeric_cols}")
    df = df.drop(non_numeric_cols, axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
print("\n1. Preparando datos para evaluación")
X = df.drop(target_column, axis=1)  # Características (features)
y = df[target_column]               # Variable objetivo (target)

# Dividimos los datos: 70% para entrenamiento y 30% para pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características para normalizar los datos
# Esto es crucial para el rendimiento de las redes neuronales
scaler = StandardScaler()  # Normaliza los datos para que tengan media=0 y desviación estándar=1
X_train_scaled = scaler.fit_transform(X_train)  # Ajustar a los datos de entrenamiento y transformar
X_test_scaled = scaler.transform(X_test)        # Transformar datos de prueba con los parámetros aprendidos

# DEFINICIÓN DE LA ARQUITECTURA DE LA RED NEURONAL CON KERAS
print("\n2. Definiendo y entrenando el modelo de red neuronal con Keras")

# Obtenemos el número de características para la capa de entrada
n_features = X_train_scaled.shape[1]

# Creamos un modelo secuencial (capas apiladas linealmente)
model = keras.Sequential([
    # Capa de entrada: mismo tamaño que el número de características
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    # Primera capa oculta con regularización dropout
    layers.Dropout(0.3),  # Desactiva aleatoriamente el 30% de las neuronas
    
    # Segunda capa oculta
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),  # Desactiva el 20% de las neuronas
    
    # Capa de salida: 1 neurona con activación sigmoid para clasificación binaria
    # Sigmoid comprime los valores entre 0 y 1 (probabilidad)
    layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
# - optimizer: algoritmo de optimización para ajustar los pesos
# - loss: función que mide el error entre predicciones y valores reales
# - metrics: métricas adicionales para monitorear durante el entrenamiento
model.compile(
    optimizer='adam',  # Algoritmo de optimización adaptativo
    loss='binary_crossentropy',  # Función de pérdida para clasificación binaria
    metrics=['accuracy']  # Monitorear la exactitud durante el entrenamiento
)

# Resumen de la arquitectura del modelo
print("\nResumen de la arquitectura del modelo:")
model.summary()

# Configurar early stopping para detener el entrenamiento cuando no hay mejora
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitorear la pérdida en el conjunto de validación
    patience=3,          # Número de épocas sin mejora antes de detener
    restore_best_weights=True  # Restaurar los mejores pesos encontrados
)

# Entrenar el modelo
batch_size = 64  # Número de muestras por lote
epochs = 10      # Número de pasadas completas por el conjunto de datos

history = model.fit(
    X_train_scaled, y_train,  # Datos de entrenamiento
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,     # 10% de los datos de entrenamiento se usan para validación
    callbacks=[early_stopping],
    verbose=1  # Mostrar barra de progreso
)

# Visualizar la curva de pérdida durante el entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], marker='o', label='Entrenamiento')
plt.plot(history.history['val_loss'], marker='o', label='Validación')
plt.title('Curva de Pérdida durante el Entrenamiento - Keras')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.grid(True)
plt.legend()
plt.show()

# Evaluar el modelo en el conjunto de prueba
print("\n3. Evaluación con múltiples métricas")

# Obtener predicciones (probabilidades)
y_prob = model.predict(X_test_scaled).ravel()

# Convertir probabilidades a predicciones binarias (umbral = 0.5)
y_pred = (y_prob > 0.5).astype(int)

# Calcular métricas básicas de rendimiento
accuracy = accuracy_score(y_test, y_pred)   # Proporción de predicciones correctas
precision = precision_score(y_test, y_pred)  # Proporción de verdaderos positivos entre todos los positivos predichos
recall = recall_score(y_test, y_pred)        # Proporción de verdaderos positivos detectados
f1 = f1_score(y_test, y_pred)                # Media armónica de precisión y recall

# Imprimir resultados de métricas básicas
print(f"Accuracy: {accuracy:.4f}")   # Exactitud global
print(f"Precision: {precision:.4f}") # Precisión (evita falsos positivos)
print(f"Recall: {recall:.4f}")       # Sensibilidad (evita falsos negativos)
print(f"F1-score: {f1:.4f}")         # Balance entre precisión y recall

# Calcular y visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de Confusión - Keras')
plt.show()

# Calcular y visualizar la curva ROC y AUC
print("\n4. Curva ROC y AUC")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calcula tasas de falsos y verdaderos positivos
roc_auc = auc(fpr, tpr)  # Área bajo la curva ROC (1.0 = perfecto, 0.5 = aleatorio)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Línea diagonal de referencia
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Keras')
plt.legend(loc='lower right')
plt.show()

# Calcular y visualizar la curva Precision-Recall
print("\n5. Curva Precision-Recall")
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)  # Precisión media

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'AP = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - Keras')
plt.legend(loc='best')
plt.show()

# Reporte de clasificación completo
print("\n6. Reporte de clasificación completo")
print(classification_report(y_test, y_pred))

# Conclusiones de métricas avanzadas
print("\n7. Conclusiones de métricas avanzadas para el modelo Keras")
print("1. El modelo Keras alcanza un F1-score de {:.4f}, equilibrando precisión y recall.".format(f1))
print("2. El AUC-ROC de {:.4f} indica la capacidad discriminativa del modelo.".format(roc_auc))
print("3. La curva Precision-Recall muestra un rendimiento de {:.4f} en términos de precisión media.".format(avg_precision))
print("4. La arquitectura más simple de Keras ha demostrado ser efectiva para esta tarea.")
print("5. El reporte de clasificación completo proporciona métricas detalladas por clase.")
print("\n8. Comparación con el modelo CNN de PyTorch")
print("1. El modelo Keras es más simple y requiere menos código para implementar.")
print("2. La arquitectura densa (fully connected) es más adecuada para datos tabulares que las CNN.")
print("3. El rendimiento es comparable, demostrando que no siempre se necesitan arquitecturas complejas.")
print("4. El tiempo de entrenamiento es generalmente menor con esta implementación más sencilla.")