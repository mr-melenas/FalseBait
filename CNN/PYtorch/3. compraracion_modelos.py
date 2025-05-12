# Comparación de modelos para la detección de phishing
# RandomForest vs MLP vs CNN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Convertir datos a tensores de PyTorch para modelos neuronales
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

# Preparar datos para CNN
feature_size = X_train_scaled.shape[1]
height = int(np.sqrt(feature_size))  # Aproximamos a un cuadrado
width = feature_size // height
if height * width < feature_size:
    width += 1

# Rellenar con ceros si es necesario
padded_size = height * width
X_train_padded = np.zeros((X_train_scaled.shape[0], padded_size))
X_test_padded = np.zeros((X_test_scaled.shape[0], padded_size))

X_train_padded[:, :feature_size] = X_train_scaled
X_test_padded[:, :feature_size] = X_test_scaled

# Reorganizar para CNN (batch_size, channels=1, height, width)
X_train_cnn = X_train_padded.reshape(-1, 1, height, width)
X_test_cnn = X_test_padded.reshape(-1, 1, height, width)

# Convertir a tensores de PyTorch para CNN
X_train_tensor_cnn = torch.FloatTensor(X_train_cnn)
X_test_tensor_cnn = torch.FloatTensor(X_test_cnn)

# Crear conjuntos de datos y cargadores de datos para modelos neuronales
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataset_cnn = TensorDataset(X_train_tensor_cnn, y_train_tensor)
test_dataset_cnn = TensorDataset(X_test_tensor_cnn, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=batch_size)

# Definir la arquitectura de la red neuronal MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Definir la arquitectura de la red neuronal CNN
class CNNClassifier(nn.Module):
    def __init__(self, height, width):
        super(CNNClassifier, self).__init__()
        
        # Primera capa convolucional
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Calcular el tamaño después de las capas convolucionales y de pooling
        conv_output_height = height // 4
        conv_output_width = width // 4
        
        # Asegurarse de que las dimensiones sean al menos 1
        conv_output_height = max(1, conv_output_height)
        conv_output_width = max(1, conv_output_width)
        
        # Calcular el tamaño de entrada para la capa fully connected
        self.fc_input_size = 32 * conv_output_height * conv_output_width
        
        # Capas fully connected
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Capas convolucionales
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Aplanar para las capas fully connected
        x = x.view(-1, self.fc_input_size)
        
        # Capas fully connected
        x = self.dropout1(self.relu3(self.fc1(x)))
        x = self.dropout2(self.relu4(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        
        return x

# Función para entrenar modelos neuronales
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

# Función para evaluar modelos neuronales
def evaluate_neural_model(model, test_loader):
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

# Inicializar modelos
print("\n2. Inicializando modelos")

# RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# MLP
input_size = X_train_scaled.shape[1]
mlp_model = MLPClassifier(input_size)
mlp_criterion = nn.BCELoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# CNN
cnn_model = CNNClassifier(height, width)
cnn_criterion = nn.BCELoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Entrenar modelos
print("\n3. Entrenando modelos")

# RandomForest
print("\n3.1 Entrenando modelo RandomForest")
rf_model.fit(X_train_scaled, y_train)

# MLP
print("\n3.2 Entrenando modelo MLP")
epochs = 10
mlp_losses = train_model(mlp_model, train_loader, mlp_criterion, mlp_optimizer, epochs)

# CNN
print("\n3.3 Entrenando modelo CNN")
cnn_losses = train_model(cnn_model, train_loader_cnn, cnn_criterion, cnn_optimizer, epochs)

# Evaluar modelos
print("\n4. Evaluando modelos")

# RandomForest
rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# MLP
mlp_true, mlp_pred, mlp_prob = evaluate_neural_model(mlp_model, test_loader)

# CNN
cnn_true, cnn_pred, cnn_prob = evaluate_neural_model(cnn_model, test_loader_cnn)

# Calcular métricas para cada modelo
models = {
    'RandomForest': {'pred': rf_pred, 'prob': rf_pred_proba},
    'MLP': {'pred': mlp_pred, 'prob': mlp_prob},
    'CNN': {'pred': cnn_pred, 'prob': cnn_prob}
}

metrics = {}
for model_name, model_data in models.items():
    if model_name == 'RandomForest':
        true_values = y_test
    else:
        true_values = mlp_true  # Ambos modelos neuronales tienen las mismas etiquetas verdaderas
    
    metrics[model_name] = {
        'accuracy': accuracy_score(true_values, model_data['pred']),
        'precision': precision_score(true_values, model_data['pred']),
        'recall': recall_score(true_values, model_data['pred']),
        'f1': f1_score(true_values, model_data['pred']),
        'auc': auc(roc_curve(true_values, model_data['prob'])[0], roc_curve(true_values, model_data['prob'])[1]),
        'avg_precision': average_precision_score(true_values, model_data['prob'])
    }

# Crear DataFrame para comparar métricas
metrics_df = pd.DataFrame({
    'Modelo': list(metrics.keys()),
    'Accuracy': [metrics[model]['accuracy'] for model in metrics],
    'Precision': [metrics[model]['precision'] for model in metrics],
    'Recall': [metrics[model]['recall'] for model in metrics],
    'F1-Score': [metrics[model]['f1'] for model in metrics],
    'AUC-ROC': [metrics[model]['auc'] for model in metrics],
    'Avg Precision': [metrics[model]['avg_precision'] for model in metrics]
})

# Mostrar tabla de comparación
print("\n5. Comparación de métricas entre modelos")
print(metrics_df.to_string(index=False))

# Visualizar comparación de métricas
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Avg Precision']
metrics_df_melted = pd.melt(metrics_df, id_vars=['Modelo'], value_vars=metrics_to_plot, var_name='Métrica', value_name='Valor')

plt.figure(figsize=(15, 10))
sns.barplot(x='Métrica', y='Valor', hue='Modelo', data=metrics_df_melted)
plt.title('Comparación de Métricas entre Modelos')
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('comparacion_metricas.png')
plt.show()

# Visualizar curvas ROC para todos los modelos
plt.figure(figsize=(10, 8))

colors = ['darkorange', 'green', 'blue']
for i, (model_name, model_data) in enumerate(models.items()):
    if model_name == 'RandomForest':
        true_values = y_test
    else:
        true_values = mlp_true
    
    fpr, tpr, _ = roc_curve(true_values, model_data['prob'])
    roc_auc = metrics[model_name]['auc']
    
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Comparación de Curvas ROC')
plt.legend(loc='lower right')
plt.savefig('comparacion_roc.png')
plt.show()

# Visualizar curvas Precision-Recall para todos los modelos
plt.figure(figsize=(10, 8))

for i, (model_name, model_data) in enumerate(models.items()):
    if model_name == 'RandomForest':
        true_values = y_test
    else:
        true_values = mlp_true
    
    precision_curve, recall_curve, _ = precision_recall_curve(true_values, model_data['prob'])
    avg_precision = metrics[model_name]['avg_precision']
    
    plt.plot(recall_curve, precision_curve, color=colors[i], lw=2, label=f'{model_name} (AP = {avg_precision:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Comparación de Curvas Precision-Recall')
plt.legend(loc='best')
plt.savefig('comparacion_precision_recall.png')
plt.show()

# Visualizar matrices de confusión para todos los modelos
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (model_name, model_data) in enumerate(models.items()):
    if model_name == 'RandomForest':
        true_values = y_test
    else:
        true_values = mlp_true
    
    cm = confusion_matrix(true_values, model_data['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
    axes[i].set_xlabel('Predicción')
    axes[i].set_ylabel('Valor real')
    axes[i].set_title(f'Matriz de Confusión - {model_name}')

plt.tight_layout()
plt.savefig('comparacion_confusion_matrices.png')
plt.show()

# Visualizar curvas de pérdida para modelos neuronales
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), mlp_losses, marker='o', color='green', label='MLP')
plt.plot(range(1, epochs+1), cnn_losses, marker='s', color='blue', label='CNN')
plt.title('Comparación de Curvas de Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.grid(True)
plt.legend()
plt.savefig('comparacion_loss_curves.png')
plt.show()

# Conclusiones
print("\n6. Conclusiones de la comparación de modelos")
print("1. Comparación de rendimiento: Se han evaluado tres modelos diferentes (RandomForest, MLP y CNN) para la detección de phishing.")
print("2. Métricas de evaluación: Se han utilizado múltiples métricas (Accuracy, Precision, Recall, F1-Score, AUC-ROC, Avg Precision) para una evaluación completa.")
print("3. Visualizaciones: Se han generado visualizaciones comparativas de curvas ROC, Precision-Recall, matrices de confusión y curvas de pérdida.")
print("4. Mejor modelo: El modelo con mejor rendimiento general es", metrics_df.iloc[metrics_df['F1-Score'].argmax()]['Modelo'], "con un F1-Score de", metrics_df['F1-Score'].max(), ".")
print("5. Consideraciones adicionales: Además del rendimiento, se deben considerar factores como tiempo de entrenamiento, interpretabilidad y requisitos computacionales.")