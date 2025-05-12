# Implementación de redes neuronales para la detección de phishing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Configuración de semilla para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# Cargar el dataset
print("Cargando el dataset...")
df = pd.read_csv("https://raw.githubusercontent.com/juancmacias/datas/main/DataSet/PhiUSIIL_Phishing_URL_Dataset.csv")
print(f"Dataset cargado correctamente. Dimensiones: {df.shape}")

# Identificar la columna objetivo
target_column = 'label'

# Identificar columnas no numéricas que podrían causar problemas
non_numeric_cols = []
for col in df.columns:
    if col != target_column:  # Excluir la columna objetivo
        try:
            df[col].astype(float)
        except Exception as e:
            non_numeric_cols.append(col)
            print(f"La columna '{col}' contiene valores no numéricos y será eliminada.")

# Eliminar columnas no numéricas
if non_numeric_cols:
    print(f"\nEliminando {len(non_numeric_cols)} columnas no numéricas: {non_numeric_cols}")
    df = df.drop(non_numeric_cols, axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
print("\n1. Preparando datos para evaluación")
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir datos a tensores de PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

# Crear conjuntos de datos y cargadores de datos
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Definir la arquitectura de la red neuronal MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        #Arquitectura de 4 capas (input → 128 → 64 → 32 → 1)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        #Durante el entrenamiento, desactiva aleatoriamente el 20% de las neuronas en cada pasada
        self.dropout = nn.Dropout(0.2) #define un dropout con probabilidad del 20%
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #El dropout se aplica en las 2 primeras capas
        #el relu (Rectified Linear Unit) se aplica en las 3 primeras capas, 
            # (Reduce el problema del desvanecimiento del gradiente, computacionalmente eficiente)
        x = self.relu(self.fc1(x))
        x = self.dropout(x) 
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x)) 
        x = self.sigmoid(self.fc4(x))  #Ideal para problemas de clasificación binaria como detección de phishing

        return x

# Inicializar el modelo, función de pérdida y optimizador
input_size = X_train_scaled.shape[1]
model = MLPClassifier(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función para entrenar el modelo
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Convertir etiquetas a formato adecuado para BCELoss
            labels = labels.view(-1, 1)
            
            # Poner a cero los gradientes
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass y optimización
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        print(f'Época {epoch+1}/{epochs}, Pérdida: {epoch_loss:.4f}')
    
    return losses

# Función para evaluar el modelo
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs.view(-1) > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(outputs.view(-1).cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# Entrenar el modelo
print("\n2. Entrenando modelo de red neuronal MLP")
epochs = 10
losses = train_model(model, train_loader, criterion, optimizer, epochs)

# Visualizar la curva de pérdida
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), losses, marker='o')
plt.title('Curva de Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.grid(True)
plt.savefig('loss_curve_nn.png')
plt.show()

# Evaluar el modelo
print("\n3. Evaluación con múltiples métricas")
y_true, y_pred, y_prob = evaluate_model(model, test_loader)

# Calcular métricas básicas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Imprimir resultados
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Calcular y visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de Confusión - Red Neuronal')
plt.savefig('confusion_matrix_nn.png')
plt.show()

# Calcular y visualizar la curva ROC y AUC
print("\n4. Curva ROC y AUC")
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Red Neuronal')
plt.legend(loc='lower right')
plt.savefig('roc_curve_nn.png')
plt.show()

# Calcular y visualizar la curva Precision-Recall
print("\n5. Curva Precision-Recall")
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
avg_precision = average_precision_score(y_true, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'AP = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - Red Neuronal')
plt.legend(loc='best')
plt.savefig('precision_recall_curve_nn.png')
plt.show()

# Reporte de clasificación completo
print("\n6. Reporte de clasificación completo")
print(classification_report(y_true, y_pred))

# Conclusiones de métricas avanzadas
#7. Conclusiones de métricas avanzadas para la red neuronal

#1. El modelo de red neuronal alcanza un F1-score de {:.4f}, equilibrando precisión y recall.".format(f1)
#2. El AUC-ROC de {:.4f} indica la capacidad discriminativa del modelo.".format(roc_auc)
#3. La curva Precision-Recall muestra un rendimiento de {:.4f} en términos de precisión media.".format(avg_precision)
#4. La arquitectura MLP con 4 capas y dropout ha demostrado ser efectiva para esta tarea.
#5. El reporte de clasificación completo proporciona métricas detalladas por clase.





