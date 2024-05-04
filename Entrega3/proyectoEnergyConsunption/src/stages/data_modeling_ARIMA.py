import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import numpy as np
from src.stages.data_ingestion import *
from src.stages.data_processing import *
from src.stages.data_split import *


# Ajustar un modelo ARIMA a los datos de entrenamiento
model_arima_zone1 = auto_arima(y_train1_zone1, seasonal=False, m=12, stepwise=True)

# Realizar predicciones en el conjunto de validación
arima_predictions_zone1 = model_arima_zone1.predict(len(y_val_zone1))

# Calcular el error cuadrático medio (MSE) de las predicciones
mse_arima_zone1 = np.mean((arima_predictions_zone1 - y_val_zone1)**2)

# Imprimir el MSE
print("MSE del modelo ARIMA en la zona 1:", mse_arima_zone1)

# Guardar el modelo
with open('/content/drive/MyDrive/0-Actual/MLOps-Bootcamp/ProyectoIndividual/models/arima_model_zone1.pkl', 'wb') as file:
  pickle.dump(model_arima_zone1, file)
