import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from src.stages.data_split import *

# Definir las funciones de transformación personalizadas
def extract_date_features(X):
    # Convertir la columna de fecha y hora a tipo datetime
    X['date_time'] = pd.to_datetime(X['date_time'])
    # Extraer características de la fecha y hora
    X['year'] = X['date_time'].dt.year
    X['month'] = X['date_time'].dt.month
    X['day'] = X['date_time'].dt.day
    X['hour'] = X['date_time'].dt.hour
    X['minute'] = X['date_time'].dt.minute
    # Eliminar la columna original de fecha y hora
    X.drop(columns=['date_time'], inplace=True)
    return X

# Definir el transformador para fechas
date_transformer = Pipeline(steps=[
    ('date_extraction', FunctionTransformer(extract_date_features)),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))  # Imputar valores faltantes (en caso de que los haya)
])

# Definir el transformador para características numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputar valores faltantes con la media
    ('scaler', StandardScaler())  # Escalar los datos
])

# Definir las columnas por tipo de datos
datetime_cols = ['date_time']
numeric_cols = ['temperature', 'humidity', 'wind_speed', 'general_diffuse_flows', 'diffuse_flows']
target_cols = ['zone1', 'zone2', 'zone3']

# Combinar los transformadores utilizando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('date', date_transformer, datetime_cols),
        ('numeric', numeric_transformer, numeric_cols)
    ]
)

# Aplicar preprocesamiento a los datos de cada zona
X_train_zone1_preprocessed = preprocessor.fit_transform(X_train1_zone1)
X_val_zone1_preprocessed = preprocessor.transform(X_val_zone1)

X_train_zone2_preprocessed = preprocessor.fit_transform(X_train1_zone2)
X_val_zone2_preprocessed = preprocessor.transform(X_val_zone2)

X_train_zone3_preprocessed = preprocessor.fit_transform(X_train1_zone3)
X_val_zone3_preprocessed = preprocessor.transform(X_val_zone3)

X_test_zone1_preprocessed = preprocessor.transform(X_test_zone1)
X_test_zone2_preprocessed = preprocessor.transform(X_test_zone2)
X_test_zone3_preprocessed = preprocessor.transform(X_test_zone3)


# Preprocesador de zona 1
with open('/content/drive/MyDrive/0-Actual/MLOps-Bootcamp/ProyectoIndividual/models/preprocessor_zone1.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)

# Preprocesador de zona 2
with open('/content/drive/MyDrive/0-Actual/MLOps-Bootcamp/ProyectoIndividual/models/preprocessor_zone2.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)

# Preprocesador de zona 3
with open('/content/drive/MyDrive/0-Actual/MLOps-Bootcamp/ProyectoIndividual/models/preprocessor_zone3.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)
    