# 1. DIO - IMPORTS I OSNOVNE POSTAVKE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')

# Učitavanje California Housing skupa podataka
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target

def remove_outliers(df, n_std=3):
    """
    Uklanja outliere iz podataka koristeći standardnu devijaciju
    """
    df_clean = df.copy()
    for column in df_clean.columns:
        mean = df_clean[column].mean()
        std = df_clean[column].std()
        df_clean = df_clean[abs(df_clean[column] - mean) <= (n_std * std)]
    return df_clean

def create_features(df):
    """
    Kreira nove značajke iz postojećih podataka
    """
    df_new = df.copy()
    
    # Interakcije između značajki
    df_new['rooms_per_household'] = df_new['AveRooms'] / df_new['AveOccup']
    df_new['bedrooms_per_room'] = df_new['AveBedrms'] / df_new['AveRooms']
    df_new['population_per_household'] = df_new['Population'] / df_new['AveOccup']
    
    # Geografske značajke
    df_new['location'] = df_new['Latitude'] * df_new['Longitude']
    
    # Log transformacije
    for col in ['MedInc', 'Population', 'AveOccup']:
        df_new[f'log_{col}'] = np.log1p(df_new[col])
        
    return df_new

def generate_detailed_report(model, X_test, y_test, y_pred, feature_importance=None):
    """
    Generira detaljni izvještaj o performansama modela
    """
    print("\n=== DETALJNI IZVJEŠTAJ O PROCJENI ===\n")
    print("Metrike točnosti modela:")
    print(f"R2 score: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# Priprema podataka
data_clean = remove_outliers(data)
data_engineered = create_features(data_clean)

# Priprema podataka za modeliranje
X = data_clean.drop('MedHouseVal', axis=1)
y = data_clean['MedHouseVal']

# Podjela podataka i skaliranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definiranje modela
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

def predict_house_price(model, features):
    """
    Predviđa cijenu kuće na temelju zadanih značajki
    """
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    return prediction * 100000  # Vraćamo u originalnu skalu

def evaluate_price(predicted_price, actual_price):
    """
    Uspoređuje predviđenu i stvarnu cijenu
    """
    difference = abs(predicted_price - actual_price) / actual_price * 100
    if difference <= 10:
        return "Cijena je realna (unutar 10% od procijenjene vrijednosti)"
    elif predicted_price > actual_price:
        return f"Cijena je potencijalno preniska. Predviđena vrijednost je {difference:.1f}% veća"
    else:
        return f"Cijena je potencijalno previsoka. Predviđena vrijednost je {difference:.1f}% manja"

# Inicijalizacija rječnika za rezultate
results = {}

# Treniranje i evaluacija modela
for name, model in models.items():
    print(f"\nTreniranje {name} modela...")
    
    # Treniranje modela
    model.fit(X_train_scaled, y_train)
    
    # Predviđanje
    y_pred = model.predict(X_test_scaled)
    
    # Izračun metrika
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Spremanje rezultata
    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    
    print(f"\nRezultati za {name}:")
    print(f"R2 score: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    # Generiranje detaljnog izvještaja
    generate_detailed_report(model, X_test_scaled, y_test, y_pred)

# Odabir najboljeg modela
best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
best_model = models[best_model_name]

print(f"\nNajbolji model: {best_model_name}")
print(f"R2 score: {results[best_model_name]['R2']:.3f}")
print(f"MAE: {results[best_model_name]['MAE']:.3f}")
print(f"RMSE: {results[best_model_name]['RMSE']:.3f}")

# Spremanje najboljeg modela
joblib.dump(best_model, 'best_housing_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Unos korisničkih vrijednosti i predviđanje
print("\nUnesite vrijednosti za predviđanje cijene kuće:")
test_features = []
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                 'AveOccup', 'Latitude', 'Longitude']

for feature in feature_names:
    while True:
        try:
            value = float(input(f"Unesite {feature}: "))
            test_features.append(value)
            break
        except ValueError:
            print("Molimo unesite validan broj.")

predicted_price = predict_house_price(best_model, test_features)
print(f"\nPredviđena cijena: ${predicted_price:.2f}")
