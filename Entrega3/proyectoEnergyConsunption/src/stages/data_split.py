import pandas as pd
import numpy as np
from src.stages.data_ingestion import *

# Dividir los datos en características (X) y objetivo (y)
X = pre_df[['date_time','temperature', 'humidity', 'wind_speed', 'general_diffuse_flows', 'diffuse_flows']]
y_zone1 = pre_df['zone1']
y_zone2 = pre_df['zone2']
y_zone3 = pre_df['zone3']

# Dividir los datos en conjunto de entrenamiento y prueba para cada zona
X_train_zone1, X_test_zone1, y_train_zone1, y_test_zone1 = train_test_split(X, y_zone1, test_size=0.2, random_state=42)
X_train_zone2, X_test_zone2, y_train_zone2, y_test_zone2 = train_test_split(X, y_zone2, test_size=0.2, random_state=42)
X_train_zone3, X_test_zone3, y_train_zone3, y_test_zone3 = train_test_split(X, y_zone3, test_size=0.2, random_state=42)

# Train and validation data

X_train1_zone1, X_val_zone1, y_train1_zone1, y_val_zone1 = train_test_split(X_train_zone1, y_train_zone1, test_size=0.2, random_state=42)
X_train1_zone2, X_val_zone2, y_train1_zone2, y_val_zone2 = train_test_split(X_train_zone2, y_train_zone2, test_size=0.2, random_state=42)
X_train1_zone3, X_val_zone3, y_train1_zone3, y_val_zone3 = train_test_split(X_train_zone3, y_train_zone3, test_size=0.2, random_state=42)

