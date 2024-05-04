import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from src.stages.data_ingestion import *
from src.stages.data_processing import *
from src.stages.data_split import *

def train_and_evaluate_model(X_train, y_train, X_val, y_val):
  """
  Trains a random forest model and calculates performance metrics.

  Args:
    X_train: Training features.
    y_train: Training target.
    X_val: Validation features.
    y_val: Validation target.

  Returns:
    model: Trained random forest model.
    mse: Mean squared error on the validation set.
    mae: Mean absolute error on the validation set.
    rmse: Root mean squared error on the validation set.
  """

  # Train the model
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  # Make predictions
  predictions = model.predict(X_val)

  # Calculate performance metrics
  mse = metrics.mean_squared_error(y_val, predictions)
  mae = metrics.mean_absolute_error(y_val, predictions)
  rmse = np.sqrt(mse)

  return model, mse, mae, rmse

# Train and evaluate models for each zone
model_zone1, mse_zone1, mae_zone1, rmse_zone1 = train_and_evaluate_model(X_train_zone1_preprocessed, y_train1_zone1, X_val_zone1_preprocessed, y_val_zone1)
model_zone2, mse_zone2, mae_zone2, rmse_zone2 = train_and_evaluate_model(X_train_zone2_preprocessed, y_train1_zone2, X_val_zone2_preprocessed, y_val_zone2)
model_zone3, mse_zone3, mae_zone3, rmse_zone3 = train_and_evaluate_model(X_train_zone3_preprocessed, y_train1_zone3, X_val_zone3_preprocessed, y_val_zone3)

# Print the results
print("Zone 1:")
print("MSE:", mse_zone1)
print("MAE:", mae_zone1)
print("RMSE:", rmse_zone1)

print("\nZone 2:")
print("MSE:", mse_zone2)
print("MAE:", mae_zone2)
print("RMSE:", rmse_zone2)

print("\nZone 3:")
print("MSE:", mse_zone3)
print("MAE:", mae_zone3)
print("RMSE:", rmse_zone3)

# Save the models
with open('/content/drive/MyDrive/0-Actual/MLOps-Bootcamp/ProyectoIndividual/models/random_forest_model_zone1.pkl', 'wb') as file:
  pickle.dump(model_zone1, file)

with open('/content/drive/MyDrive/0-Actual/MLOps-Bootcamp/ProyectoIndividual/models/random_forest_model_zone2.pkl', 'wb') as file:
  pickle.dump(model_zone2, file)

with open('/content/drive/MyDrive/0-Actual/MLOps-Bootcamp/ProyectoIndividual/models/random_forest_model_zone3.pkl', 'wb') as file:
  pickle.dump(model_zone3, file)


def analyze_zone_predictions(model, X_test, y_test, X_test_preprocessed, zone_name):
    # Realizar predicciones en el conjunto de test
    predictions = model.predict(X_test_preprocessed)

    # Calcular el error cuadrático medio (MSE) de las predicciones
    mse = np.mean((predictions - y_test)**2)

    # Imprimir el MSE
    print(f"MSE del modelo Random Forest en la {zone_name}:", mse)

    # Crear un DataFrame con las predicciones y los valores reales
    df_predictions = pd.DataFrame({
        'date_time': X_test['date_time'],
        'actual': y_test,
        'prediction': predictions
    })

    # Agrupar por mes y calcular el promedio
    df_predictions_monthly = df_predictions.groupby(df_predictions['date_time'].dt.month).mean()

    # Crear un gráfico de líneas
    plt.figure(figsize=(10, 6))
    plt.plot(df_predictions_monthly['date_time'], df_predictions_monthly['actual'], label='Actual')
    plt.plot(df_predictions_monthly['date_time'], df_predictions_monthly['prediction'], label='Prediction')
    plt.xlabel('Month')
    plt.ylabel(f'{zone_name} Power Consumption')
    plt.title(f'Random Forest Predictions vs. Actual Values ({zone_name})')
    plt.legend()
    plt.show()
    
# ### Zone1
# Uso de la función
analyze_zone_predictions(model_zone1, X_test_zone1, y_test_zone1, X_test_zone1_preprocessed, "Zone 1")

# ### Zone2
# Uso de la función
analyze_zone_predictions(model_zone2, X_test_zone2, y_test_zone2, X_test_zone2_preprocessed, "Zone 2")


# ### Zone3
# Uso de la función
analyze_zone_predictions(model_zone3, X_test_zone3, y_test_zone3, X_test_zone3_preprocessed, "Zone 3")