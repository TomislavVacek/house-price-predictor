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

# Ispis osnovnih informacija
print("Prvih pet redova podataka:")
print(data.head())
print("\nInformacije o skupu podataka:")
print(data.info())

# Funkcija za uklanjanje outliera s poboljšanim pristupom
def remove_outliers(df, n_std=3):
    df_clean = df.copy()
    for column in df_clean.columns:
        mean = df_clean[column].mean()
        std = df_clean[column].std()
        df_clean = df_clean[abs(df_clean[column] - mean) <= (n_std * std)]
    return df_clean

# Feature Engineering
def create_features(df):
    df_new = df.copy()
    
    # Interakcije između značajki
    df_new['rooms_per_household'] = df_new['AveRooms'] / df_new['AveOccup']
    df_new['bedrooms_per_room'] = df_new['AveBedrms'] / df_new['AveRooms']
    df_new['population_per_household'] = df_new['Population'] / df_new['AveOccup']
    
    # Geografske značajke
    df_new['location'] = df_new['Latitude'] * df_new['Longitude']
    df_new['location_cluster'] = df_new['location'].apply(lambda x: round(x, 1))
    
    # Log transformacije
    for col in ['MedInc', 'Population', 'AveOccup']:
        df_new[f'log_{col}'] = np.log1p(df_new[col])
        
    return df_new

# Priprema podataka
data_clean = remove_outliers(data)
data_engineered = create_features(data_clean)
print(f"\nVeličina podataka prije uklanjanja outliera: {len(data)}")
print(f"Veličina podataka nakon uklanjanja outliera: {len(data_clean)}")

# Priprema podataka za modeliranje
X = data_engineered.drop(['MedHouseVal', 'location_cluster'], axis=1)
y = data_engineered['MedHouseVal']

# Standardizacija i podjela podataka
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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

# Treniranje i evaluacija modela
results = {}
for name, model in models.items():
    print(f"\nTreniranje {name} modela...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    print(f"\nRezultati {name} modela:")
    print(f"R2 score: {results[name]['R2']:.3f}")
    print(f"MAE: {results[name]['MAE']:.3f}")
    print(f"MSE: {results[name]['MSE']:.3f}")
    print(f"RMSE: {results[name]['RMSE']:.3f}")
    
    # Cross-validacija
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"\nCross-validation rezultati za {name}:")
    print(f"Scores: {cv_scores}")
    print(f"Prosječni CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

def predict_house_price(model, scaler, features):
    """
    Funkcija koja predviđa cijenu kuće na temelju unesenih karakteristika.
    """
    # Pretvaranje unosa u DataFrame
    input_data = pd.DataFrame([features])
    
    # Kreiranje dodatnih značajki kao u treningu
    input_data['rooms_per_household'] = input_data['AveRooms'] / input_data['AveOccup']
    input_data['bedrooms_per_room'] = input_data['AveBedrms'] / input_data['AveRooms']
    input_data['population_per_household'] = input_data['Population'] / input_data['AveOccup']
    input_data['location'] = input_data['Latitude'] * input_data['Longitude']
    
    # Log transformacije
    for col in ['MedInc', 'Population', 'AveOccup']:
        input_data[f'log_{col}'] = np.log1p(input_data[col])
    
    # Skaliranje podataka
    input_scaled = scaler.transform(input_data)
    
    # Predikcija
    predicted_price = model.predict(input_scaled)[0]
    
    return predicted_price

def evaluate_price(predicted_price, actual_price):
    """
    Uspoređuje predviđenu i stvarnu cijenu i daje procjenu.
    """
    difference = abs(predicted_price - actual_price) / actual_price * 100
    
    if difference <= 10:
        return "Cijena je realna (unutar 10% od procijenjene vrijednosti)"
    elif predicted_price > actual_price:
        return f"Cijena je potencijalno preniska. Predviđena vrijednost je {difference:.1f}% veća"
    else:
        return f"Cijena je potencijalno previsoka. Predviđena vrijednost je {difference:.1f}% manja"

# Spremanje najboljih modela i skalera
joblib.dump(models['Gradient Boosting'], 'house_price_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Interaktivno sučelje za predviđanje cijena
def predict_interface():
    print("\n=== PROCJENITELJ VRIJEDNOSTI NEKRETNINE ===")
    
    try:
        features = {
            'MedInc': float(input("Unesite prosječni prihod u području (u 10000$): ")),
            'HouseAge': float(input("Unesite starost kuće (godine): ")),
            'AveRooms': float(input("Unesite prosječan broj soba: ")),
            'AveBedrms': float(input("Unesite broj spavaćih soba: ")),
            'Population': float(input("Unesite populaciju područja: ")),
            'AveOccup': float(input("Unesite prosječnu popunjenost: ")),
            'Latitude': float(input("Unesite geografsku širinu: ")),
            'Longitude': float(input("Unesite geografsku dužinu: "))
        }
        
        actual_price = float(input("\nUnesite stvarnu cijenu nekretnine (u 100000$): "))
        
        # Korištenje Gradient Boosting modela za predikciju
        predicted_price = predict_house_price(models['Gradient Boosting'], scaler, features)
        
        print("\n=== REZULTATI PROCJENE ===")
        print(f"Predviđena cijena: ${predicted_price*100000:.2f}")
        print(f"Stvarna cijena: ${actual_price*100000:.2f}")
        print(evaluate_price(predicted_price, actual_price))
        
    except ValueError:
        print("Pogreška: Molimo unesite važeće brojčane vrijednosti.")
    except Exception as e:
        print(f"Došlo je do pogreške: {str(e)}")

# Vizualizacija važnosti značajki
best_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 najvažnijih značajki')
plt.tight_layout()
plt.show()

# Stvarne vs. Predviđene vrijednosti
plt.figure(figsize=(10, 6))
plt.scatter(y_test, models['Random Forest'].predict(X_test), alpha=0.5, label='Random Forest')
plt.scatter(y_test, models['Gradient Boosting'].predict(X_test), alpha=0.5, label='Gradient Boosting')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Stvarne cijene')
plt.ylabel('Predviđene cijene')
plt.title('Stvarne vs. Predviđene cijene')
plt.legend()
plt.tight_layout()
plt.show()

if __name__ == "__main__":
    # Sav postojeći kod za treniranje i evaluaciju
    
    # Pitajte korisnika želi li predvidjeti cijenu
    while True:
        choice = input("\nŽelite li predvidjeti cijenu nekretnine? (da/ne): ").lower()
        if choice == 'da':
            predict_interface()
        elif choice == 'ne':
            break
        else:
            print("Molimo odgovorite sa 'da' ili 'ne'")