import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from core.config import settings
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


load_dotenv()
# Initialize Supabase client

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def load_data():
    # Leer todos los registros de la tabla
    response = supabase.table("phishing_inputs_tld").select("*").execute()
    # Mostrar datos
    if response.data:
        try:
            # Verificamos si hay datos y los guardamos en CSV
            df = pd.DataFrame(response.data)
            df.to_csv(settings.test_data_logs, index=False)
            print("Datos guardados en CSV:", settings.test_data_logs)
            save_concat()
        except Exception as e:
            print("Error al guardar los datos en CSV:", e)
        for row in response.data:
            print(row)
    else:
        print("No se encontraron datos o hubo un error:", response.error)

# Concat CSV and save
def save_concat():
    try:
        # Leer los dos archivos
        df1 = pd.read_csv("https://raw.githubusercontent.com/juancmacias/datas/main/DataSet/PhiUSIIL_Phishing_URL_Dataset.csv")
        df2 = pd.read_csv(settings.test_data_logs)
        df_total = pd.concat([df1, df2], ignore_index=True)
        df_total.to_csv("csv/combinado_modelado.csv", index=False)
        print("Archivos combinados y guardados como combinado.csv")
    except Exception as e:
        print("Error al combinar los archivos:", e)

# Aquí puedes agregar la lógica de tu modelo
def model_b():
    try:
        df = pd.read_csv("csv/combinado_modelado.csv")
        # Limpiar columnas
        columnas_a_eliminar = ['FILENAME', 'URL', 'Domain', 'Title']

        df.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')
        # guardar mapeo
        tld_mapping = dict(enumerate(df['TLD'].astype('category').cat.categories))
        reverse_tld_mapping = {v: k for k, v in tld_mapping.items()}
        #joblib.dump(reverse_tld_mapping, "tld_mapping.pkl")

        # Codificar TLD como entero
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar modelo RandomforestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Predicciones y evaluación
        y_pred = clf.predict(X_test)

        print("Reporte de clasificación:\n")
        print(classification_report(y_test, y_pred))
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

    except Exception as e:
            print("Error al combinar los archivos:", e)
