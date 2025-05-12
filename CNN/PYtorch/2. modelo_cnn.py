# Implementación de redes neuronales convolucionales (CNN) para la detección de phishing

#######################################################
# SECCIÓN DE DEFINICIONES DE CONCEPTOS CLAVE
#######################################################

# TENSORES DE PYTORCH:
# Los tensores son estructuras de datos multidimensionales similares a los arrays de NumPy,
# pero con la capacidad de ejecutarse en GPUs para acelerar los cálculos. Son la base de PyTorch
# y permiten realizar operaciones matemáticas eficientes necesarias para el deep learning.
# Un tensor puede ser un escalar (0D), vector (1D), matriz (2D) o tener más dimensiones.

# CNN (REDES NEURONALES CONVOLUCIONALES):
# Son un tipo especializado de red neuronal diseñado principalmente para procesar datos con estructura
# de cuadrícula, como imágenes. A diferencia de las redes neuronales tradicionales (MLP),
# las CNN mantienen la estructura espacial de los datos y aplican filtros (kernels) que se
# deslizan por la entrada para detectar patrones locales. Aunque normalmente se usan para imágenes,
# en este caso adaptamos nuestros datos tabulares a un formato 2D para aprovechar sus ventajas.

# CAPAS CONVOLUCIONALES:
# Son el componente principal de las CNN. Aplican operaciones de convolución utilizando filtros (kernels)
# que se deslizan por los datos de entrada para detectar características específicas. Cada filtro
# genera un mapa de características que resalta patrones particulares en los datos. Las primeras capas
# detectan características simples (como bordes) y las más profundas detectan patrones más complejos.

# MAXPOOLING:
# Es una operación de reducción de dimensionalidad que disminuye el tamaño espacial de los mapas de
# características, seleccionando el valor máximo en cada región. Esto ayuda a:
# 1. Reducir la cantidad de parámetros y cálculos en la red
# 2. Controlar el sobreajuste
# 3. Hacer que la detección de características sea más robusta a cambios de posición

# BATCH NORMALIZATION:
# Técnica que normaliza las activaciones de una capa, lo que acelera el entrenamiento
# y mejora la estabilidad de la red neuronal.

# DROPOUT:
# Técnica de regularización que desactiva aleatoriamente un porcentaje de neuronas durante
# el entrenamiento para evitar el sobreajuste (cuando el modelo memoriza los datos de entrenamiento
# en lugar de aprender patrones generalizables).

#######################################################


import pandas as pd 
import numpy as np   
import matplotlib.pyplot as plt 
import seaborn as sns 

# Bibliotecas de PyTorch para deep learning
import torch  # Framework principal de deep learning
import torch.nn as nn  # Módulos de redes neuronales
import torch.optim as optim  # Optimizadores para entrenamiento
from torch.utils.data import Dataset, DataLoader, TensorDataset  # Para manejo eficiente de datos

# Bibliotecas de scikit-learn para preprocesamiento y evaluación
from sklearn.model_selection import train_test_split  # Para dividir datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler  # Para normalizar características
from sklearn.metrics import (  # Métricas de evaluación
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')  # Suprime advertencias para una salida más limpia

# Configuración de semilla para reproducibilidad de resultados
# se utiliza 42 por convención en la comunidad de aprendizaje automático y deep learning (xq patata)
torch.manual_seed(42)  # Semilla para PyTorch
np.random.seed(42)     # Semilla para NumPy

# Cargar el dataset desde GitHub
print("Cargando el dataset...")
df = pd.read_csv("https://raw.githubusercontent.com/juancmacias/datas/main/DataSet/PhiUSIIL_Phishing_URL_Dataset.csv")
print(f"Dataset cargado correctamente. Dimensiones: {df.shape}")

# Identificar la columna objetivo (variable dependiente)
target_column = 'label'  # Etiqueta que indica si una URL es phishing (1) o legítima (0)

# Identificar y eliminar columnas no numéricas que podrían causar problemas
# Las redes neuronales requieren datos numéricos para funcionar correctamente
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

# PREPARACIÓN DE DATOS PARA CNN
# Las CNN esperan datos en formato de imagen (batch_size, channels, height, width)
# Aquí transformamos nuestros datos tabulares en un formato 2D (similar a una imagen)

# Paso 1: Determinar las dimensiones de la "imagen" que crearemos
feature_size = X_train_scaled.shape[1]  # Número de características
height = int(np.sqrt(feature_size))     # Calculamos altura aproximada para un cuadrado
width = feature_size // height          # Calculamos ancho
if height * width < feature_size:       # Ajustamos si es necesario
    width += 1

# Paso 2: Rellenar con ceros si es necesario para completar la matriz
padded_size = height * width  # Tamaño total después del relleno
X_train_padded = np.zeros((X_train_scaled.shape[0], padded_size))  # Matriz con ceros
X_test_padded = np.zeros((X_test_scaled.shape[0], padded_size))    # Matriz con ceros

# Copiamos los datos originales a las nuevas matrices rellenadas
X_train_padded[:, :feature_size] = X_train_scaled
X_test_padded[:, :feature_size] = X_test_scaled

# Paso 3: Reorganizar para CNN en formato (batch_size, channels=1, height, width)
# El canal único (1) es porque nuestros datos no tienen canales de color como RGB
X_train_cnn = X_train_padded.reshape(-1, 1, height, width)  # -1 mantiene el número de muestras
X_test_cnn = X_test_padded.reshape(-1, 1, height, width)

# Paso 4: Convertir a tensores de PyTorch
# Los tensores son el tipo de dato fundamental en PyTorch, similar a los arrays de NumPy
# pero optimizados para deep learning y compatibles con GPU
X_train_tensor = torch.FloatTensor(X_train_cnn)    # Convierte arrays de NumPy a tensores de PyTorch
X_test_tensor = torch.FloatTensor(X_test_cnn)      # FloatTensor indica que son de tipo float
y_train_tensor = torch.FloatTensor(y_train.values) # Convertimos también las etiquetas
y_test_tensor = torch.FloatTensor(y_test.values)

# Paso 5: Crear conjuntos de datos y cargadores de datos (DataLoaders)
# Los DataLoaders permiten cargar datos en lotes (batches) durante el entrenamiento
# lo que es más eficiente en memoria y computación
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # Combina características y etiquetas
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64  # Número de muestras por lote
# Los DataLoaders cargan los datos en lotes y pueden mezclarlos (shuffle) para el entrenamiento
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # shuffle=True mezcla los datos
test_loader = DataLoader(test_dataset, batch_size=batch_size)  # No mezclamos los datos de prueba

# DEFINICIÓN DE LA ARQUITECTURA DE LA RED NEURONAL CNN
class CNNClassifier(nn.Module):
    def __init__(self, height, width):
        super(CNNClassifier, self).__init__()  # Inicializa la clase base
        
        # Primera capa convolucional
        # Entrada: 1 canal (datos en escala de grises)
        # Salida: 16 mapas de características (16 filtros diferentes)
        # kernel_size=3: filtro de 3x3 píxeles
        # padding=1: añade un borde de 1 píxel para mantener las dimensiones espaciales
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  # Función de activación no lineal
        # MaxPool reduce las dimensiones a la mitad (kernel_size=2)
        # Esto ayuda a reducir la complejidad y a enfocar en características importantes
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Segunda capa convolucional
        # Entrada: 16 mapas de características (de la capa anterior)
        # Salida: 32 mapas de características (32 filtros diferentes)
        # Cada filtro aprende patrones más complejos combinando los de la capa anterior
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # Reduce dimensiones nuevamente
        
        # Calcular el tamaño después de las capas convolucionales y de pooling
        # Después de dos capas de pooling con kernel_size=2, las dimensiones se reducen a 1/4
        # Ejemplo: si empezamos con 10x10, después de pool1 será 5x5, después de pool2 será 2x2
        conv_output_height = height // 4  # División entera
        conv_output_width = width // 4
        
        # Asegurarse de que las dimensiones sean al menos 1 (evitar dimensiones de 0)
        conv_output_height = max(1, conv_output_height)
        conv_output_width = max(1, conv_output_width)
        
        # Calcular el tamaño de entrada para la capa fully connected
        # Multiplicamos: número de filtros (32) x altura x ancho
        self.fc_input_size = 32 * conv_output_height * conv_output_width
        
        # Capas fully connected (densamente conectadas)
        # Estas capas finales toman las características extraídas por las capas convolucionales
        # y las utilizan para la clasificación final
        self.fc1 = nn.Linear(self.fc_input_size, 128)  # Primera capa FC: reduce a 128 neuronas
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # Desactiva aleatoriamente el 50% de las neuronas durante el entrenamiento
        
        self.fc2 = nn.Linear(128, 64)  # Segunda capa FC: reduce a 64 neuronas
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)  # Desactiva el 30% de las neuronas
        
        self.fc3 = nn.Linear(64, 1)  # Capa de salida: 1 neurona para clasificación binaria
        self.sigmoid = nn.Sigmoid()  # Función de activación sigmoid para obtener probabilidad entre 0 y 1
        
    def forward(self, x):
        # Este método define el flujo de datos a través de la red
        
        # Paso 1: Procesamiento a través de las capas convolucionales
        # Secuencia: Conv2D -> ReLU -> MaxPool (para ambas capas)
        x = self.pool1(self.relu1(self.conv1(x)))  # Primera capa convolucional
        x = self.pool2(self.relu2(self.conv2(x)))  # Segunda capa convolucional
        
        # Paso 2: Aplanar el tensor para las capas fully connected
        # Convertimos de (batch_size, channels, height, width) a (batch_size, fc_input_size)
        x = x.view(-1, self.fc_input_size)  # -1 infiere automáticamente el tamaño del batch
        
        # Paso 3: Procesamiento a través de las capas fully connected
        # Secuencia: Linear -> ReLU -> Dropout (para las primeras dos capas FC)
        x = self.dropout1(self.relu3(self.fc1(x)))  # Primera capa FC
        x = self.dropout2(self.relu4(self.fc2(x)))  # Segunda capa FC
        x = self.sigmoid(self.fc3(x))               # Capa de salida con sigmoid
        
        return x  # Devuelve la probabilidad de que la URL sea phishing

# Inicializar el modelo, función de pérdida y optimizador
model = CNNClassifier(height, width)  # Creamos una instancia de nuestro modelo

# Binary Cross Entropy Loss: función de pérdida adecuada para clasificación binaria
# Mide la diferencia entre la probabilidad predicha y la etiqueta real
criterion = nn.BCELoss()

# Adam: optimizador adaptativo que ajusta automáticamente las tasas de aprendizaje
# lr=0.001: tasa de aprendizaje inicial
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función para entrenar el modelo
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()  # Establece el modelo en modo de entrenamiento (activa dropout, etc.)
    losses = []    # Para almacenar las pérdidas de cada época
    #- Un epoch completo significa que cada muestra en el conjunto de datos de entrenamiento ha pasado una vez por la red neuronal (tanto en la propagación hacia adelante como en la retropropagación).
    #- Durante cada epoch, el modelo ajusta sus pesos basándose en el error calculado.
    for epoch in range(epochs):
        running_loss = 0.0
        # Iteramos sobre los lotes de datos
        for inputs, labels in train_loader:
            # Convertir etiquetas a formato adecuado para BCELoss (de [batch_size] a [batch_size, 1])
            labels = labels.view(-1, 1)
            
            # Paso 1: Poner a cero los gradientes acumulados de pasos anteriores
            optimizer.zero_grad()
            
            # Paso 2: Forward pass - propagación hacia adelante
            # Pasamos los datos por la red y obtenemos predicciones
            outputs = model(inputs)
            
            # Paso 3: Calcular la pérdida (error) entre predicciones y etiquetas reales
            loss = criterion(outputs, labels)
            
            # Paso 4: Backward pass - retropropagación del error
            # Calcula los gradientes de la pérdida con respecto a los parámetros del modelo
            loss.backward()
            
            # Paso 5: Optimización - actualizar los pesos del modelo usando los gradientes
            optimizer.step()
            
            # Acumular la pérdida para esta época
            running_loss += loss.item() * inputs.size(0)
        
        # Calcular la pérdida promedio para esta época
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        print(f'Época {epoch+1}/{epochs}, Pérdida: {epoch_loss:.4f}')
    
    return losses  # Devuelve el historial de pérdidas para visualización

# Función para evaluar el modelo en el conjunto de prueba
def evaluate_model(model, test_loader):
    model.eval()  # Establece el modelo en modo de evaluación (desactiva dropout, etc.)
    
    # Listas para almacenar resultados
    y_true = []  # Etiquetas reales
    y_pred = []  # Predicciones binarias (0 o 1)
    y_prob = []  # Probabilidades predichas
    
    # torch.no_grad() desactiva el cálculo de gradientes durante la evaluación
    # lo que ahorra memoria y acelera el proceso
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Obtener predicciones del modelo
            outputs = model(inputs)
            
            # Convertir probabilidades a predicciones binarias (umbral = 0.5)
            predicted = (outputs.view(-1) > 0.5).float()
            
            # Almacenar resultados
            y_true.extend(labels.cpu().numpy())          # Etiquetas reales
            y_pred.extend(predicted.cpu().numpy())       # Predicciones binarias
            y_prob.extend(outputs.view(-1).cpu().numpy()) # Probabilidades
    
    # Convertir listas a arrays de NumPy para cálculos posteriores
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# Entrenar el modelo
print("\n2. Entrenando modelo de red neuronal CNN")
epochs = 10  # Número de pasadas completas por el conjunto de datos
losses = train_model(model, train_loader, criterion, optimizer, epochs)

# Visualizar la curva de pérdida durante el entrenamiento
# Esto nos permite ver cómo disminuye el error a lo largo del tiempo
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), losses, marker='o')
plt.title('Curva de Pérdida durante el Entrenamiento - CNN')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.grid(True)
#plt.savefig('loss_curve_cnn.png') 
plt.show()

# Evaluar el modelo en el conjunto de prueba
print("\n3. Evaluación con múltiples métricas")
y_true, y_pred, y_prob = evaluate_model(model, test_loader)

# Calcular métricas básicas de rendimiento
accuracy = accuracy_score(y_true, y_pred)   # Proporción de predicciones correctas
precision = precision_score(y_true, y_pred)  # Proporción de verdaderos positivos entre todos los positivos predichos
recall = recall_score(y_true, y_pred)        # Proporción de verdaderos positivos detectados
f1 = f1_score(y_true, y_pred)                # Media armónica de precisión y recall

# Imprimir resultados de métricas básicas
print(f"Accuracy: {accuracy:.4f}")   # Exactitud global
print(f"Precision: {precision:.4f}") # Precisión (evita falsos positivos)
print(f"Recall: {recall:.4f}")       # Sensibilidad (evita falsos negativos)
print(f"F1-score: {f1:.4f}")         # Balance entre precisión y recall

# Calcular y visualizar la matriz de confusión
# Muestra la distribución de predicciones correctas e incorrectas
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de Confusión - CNN')
#plt.savefig('confusion_matrix_cnn.png') 
plt.show()

# Calcular y visualizar la curva ROC y AUC
# La curva ROC muestra el rendimiento del clasificador a diferentes umbrales
print("\n4. Curva ROC y AUC")
fpr, tpr, thresholds = roc_curve(y_true, y_prob)  # Calcula tasas de falsos y verdaderos positivos
roc_auc = auc(fpr, tpr)  # Área bajo la curva ROC (1.0 = perfecto, 0.5 = aleatorio)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Línea diagonal de referencia
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - CNN')
plt.legend(loc='lower right')
#plt.savefig('roc_curve_cnn.png')  
plt.show()

# Calcular y visualizar la curva Precision-Recall
# Útil cuando las clases están desequilibradas
print("\n5. Curva Precision-Recall")
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
avg_precision = average_precision_score(y_true, y_prob)  # Precisión media

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'AP = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - CNN')
plt.legend(loc='best')
#plt.savefig('precision_recall_curve_cnn.png')  # Descomentar para guardar la figura
plt.show()

# Reporte de clasificación completo
# Proporciona métricas detalladas por clase
print("\n6. Reporte de clasificación completo")
print(classification_report(y_true, y_pred))

# Conclusiones de métricas avanzadas
print("\n7. Conclusiones de métricas avanzadas para la CNN")
print("1. El modelo CNN alcanza un F1-score de {:.4f}, equilibrando precisión y recall.".format(f1))
print("2. El AUC-ROC de {:.4f} indica la capacidad discriminativa del modelo.".format(roc_auc))
print("3. La curva Precision-Recall muestra un rendimiento de {:.4f} en términos de precisión media.".format(avg_precision))
print("4. La arquitectura CNN con capas convolucionales y pooling ha demostrado ser efectiva para esta tarea.")
print("5. El reporte de clasificación completo proporciona métricas detalladas por clase.")