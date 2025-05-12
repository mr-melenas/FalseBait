import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from core.config import settings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
import shutil
import joblib
import csv
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


load_dotenv()
# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Load data of SQL
async def load_data(number_itera=50):
    
    # Create CSV file
    if number_itera > 1:
        df = pd.read_csv(settings.test_data_logs)
        # Obtener la última fila de la columna 'nombre_columna'
        last_file = df['id'].iloc[-1]
        # Leer todos los registros de la tabla
        response = supabase.table("phishing_inputs_tld")\
                .select("*", count="exact")\
                .gt("id", last_file)\
                .execute()
        # Verificamos si hay datos nuevos segun el numero de iteraciones que queremos
        if  response.count >= number_itera:
            print("Se han encontrado datos nuevos en la tabla.")
            try:
                data_filtrada = [
                    {k: v for k, v in row.items() if k not in ("created_at")}
                    for row in response.data
                    ]
                
                df = pd.DataFrame(data_filtrada)
                # quitar duplicados
                #df.drop_duplicates(subset=['URL'], inplace=True)
                df.to_csv(settings.test_data_logs, index=False)
                save_concat()
            except Exception as e:
                print("Error al guardar los datos en CSV:", e)
        else    :
            print("No se encontraron datos suficientes según el número de iteraciones: ", number_itera)
            print("Contados en la tabla: ", response.count)    
    else:
        print("No se encontraron datos... ")

# Concat CSV and save
def save_concat():
    try:
        # Leer los dos archivos
        df1 = pd.read_csv("https://raw.githubusercontent.com/juancmacias/datas/main/DataSet/PhiUSIIL_Phishing_URL_Dataset.csv")
        df2 = pd.read_csv(settings.test_data_logs)
        df_total = pd.concat([df1, df2], ignore_index=True)
        df_total.to_csv(settings.combined_data, index=False)
        print("Archivos combinados y guardados como:", settings.combined_data)
        model_b()
    except Exception as e:
        print("Error al combinar los archivos:", e)

# Entrenamiento del modelo B, datos combinados
def model_b():
    try:
        df = pd.read_csv(settings.combined_data)
        
        # Limpiar columnas
        columnas_a_eliminar = ['FILENAME', 'URL', 'Domain', 'Title', 'id']
        df.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')
        # guardar mapeo
        #tld_mapping = dict(enumerate(df['TLD'].astype('category').cat.categories))
        #reverse_tld_mapping = {v: k for k, v in tld_mapping.items()}
        #joblib.dump(reverse_tld_mapping, "tld_mapping.pkl")

        # Codificar TLD 
        df['TLD'] = df['TLD'].astype('category').cat.codes

        # Asegurar booleanos como enteros
        boolean_cols = [
            'IsDomainIP','HasObfuscation','IsHTTPS','HasTitle','HasFavicon',
            'IsResponsive','HasDescription','HasExternalFormSubmit','HasSocialNet',
            'HasSubmitButton','HasHiddenFields','HasPasswordField',
            'Bank','Pay','Crypto','HasCopyrightInfo'
        ]

        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Separar features y label
        X = df.drop('label', axis=1)
        y = df['label']

        # Dividir en entrenamiento y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        clf = RandomForestClassifier(class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)

        # Predicciones y evaluación
        y_pred = clf.predict(X_test)

        # Matriz de confusión
        print("Matriz de confusión:\n")
        print(confusion_matrix(y_test, y_pred))
        # Reporte de clasificación
        print(classification_report(y_test, y_pred))
        #validate(y_test, y_pred, y_train, X_train, clf)
        # Guardar el modelo
        joblib.dump(clf, settings.model_path_B)
        print("Modelo guardado como :", settings.model_path_B)
        
        comparar_y_reemplazar_modelo(X_test, y_test, settings.model_path_A, settings.model_path_B)

    except Exception as e:
            print("Error al combinar los archivos:", e)

# Reemplazar el modelo A por el B si hay mejora
def validate(y_test, y_pred, y_train, X_train, clf):
    
    # Validación de Overfitting
    y_train_pred = clf.predict(X_train)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_pred)
    diff = abs(acc_train - acc_test)

    print(f"\nAccuracy en entrenamiento: {acc_train:.4f}")
    print(f"Accuracy en test:          {acc_test:.4f}")
    print(f"Diferencia absoluta:       {diff:.4f}")

    if diff > 0.05:
        print(f"⚠️  El modelo presenta sobreajuste > 5%", diff)
    else:
        print(f"✅  No hay sobreajuste significativo (≤ 5%)", diff)

def comparar_y_reemplazar_modelo(X_val, y_val, path_a, path_b, umbral_mejora=0.10):
    # Cargar modelos
    modelo_a = joblib.load(path_a)
    modelo_b = joblib.load(path_b)

    # Hacer predicciones
    pred_a = modelo_a.predict(X_val)
    pred_b = modelo_b.predict(X_val)

    # Calcular métricas
    f1_a = f1_score(y_val, pred_a, zero_division=1)
    f1_b = f1_score(y_val, pred_b, zero_division=1)

    print(f"F1 modelo A: {f1_a:.4f}")
    print(f"F1 modelo B: {f1_b:.4f}")
    mejora = f1_b - f1_a
    # Guardar histórico
    try:
        os.makedirs("csv", exist_ok=True)
        with open(settings.history_models, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Si el archivo está vacío, escribir encabezados
                writer.writerow(["timestamp", "f1_model_a", "f1_model_b", "improvement", "model_replaced"])
            writer.writerow([datetime.utcnow().isoformat(), f1_a, f1_b, mejora, mejora >= umbral_mejora])
    except Exception as e:
        print(f"Error al guardar el histórico: {e}")
    # Comparar mejoras
    
    if mejora >= umbral_mejora:
        shutil.copy(path_b, path_a)
        print(f"✅ Modelo B reemplazó a A (mejora de {mejora:.2%})")
        return True
    else:
        print(f"❌ No hay mejora suficiente. Se mantiene el modelo A.")
        return False
    