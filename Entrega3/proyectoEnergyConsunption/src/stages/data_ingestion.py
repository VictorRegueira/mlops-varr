import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
pre_df = pd.read_csv("G:\My Drive\0-Actual\MLOps-Bootcamp\ProyectoIndividual\data\raw\Tetuan_City_power_consumption.csv")

# Cambiar los nombres de las columnas
pre_df.rename(columns={
    'DateTime': 'date_time',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Wind Speed': 'wind_speed',
    'general diffuse flows': 'general_diffuse_flows',
    'diffuse flows': 'diffuse_flows',
    'Zone 1 Power Consumption': 'zone1',
    'Zone 2  Power Consumption': 'zone2',
    'Zone 3  Power Consumption': 'zone3'
}, inplace=True)


