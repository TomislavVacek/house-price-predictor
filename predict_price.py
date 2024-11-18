import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model():
    """Učitava spremljeni model i scaler"""
    try:
        model = joblib.load('best_housing_model.joblib')  # Promijenjeno ime datoteke
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        print("Greška: Model ili scaler datoteke nisu pronađene!")
        return None, None
    except Exception as e:
        print(f"Greška pri učitavanju modela: {str(e)}")
        return None, None

def prepare_features(features):
    """Priprema značajke za predviđanje"""
    # Koristimo samo osnovne značajke koje su korištene pri treniranju
    columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
               'AveOccup', 'Latitude', 'Longitude']
    df = pd.DataFrame([features], columns=columns)
    return df

def predict_house_price(model, scaler, features):
    """Predviđa cijenu kuće na temelju značajki"""
    try:
        input_data = prepare_features(features)
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]
        return predicted_price
    except Exception as e:
        print(f"Greška pri predviđanju: {str(e)}")
        return None

def format_price(price):
    """Formatira cijenu za prikaz"""
    return f"${price*100000:,.2f}"

def evaluate_price(predicted_price, actual_price):
    """Evaluira razliku između predviđene i stvarne cijene"""
    difference = abs(predicted_price - actual_price) / actual_price * 100
    
    if difference <= 10:
        return "✓ Cijena je realna (unutar 10% od procijenjene vrijednosti)"
    elif predicted_price > actual_price:
        return f"↑ Cijena je potencijalno preniska. Predviđena vrijednost je {difference:.1f}% veća"
    else:
        return f"↓ Cijena je potencijalno previsoka. Predviđena vrijednost je {difference:.1f}% manja"

def get_user_input():
    """Prikuplja podatke od korisnika"""
    print("\n=== UNESITE PODATKE O NEKRETNINI ===")
    try:
        features = {
            'MedInc': float(input("Prosječni prihod u području (u 10000$): ")),
            'HouseAge': float(input("Starost kuće (godine): ")),
            'AveRooms': float(input("Prosječan broj soba: ")),
            'AveBedrms': float(input("Broj spavaćih soba: ")),
            'Population': float(input("Populacija područja: ")),
            'AveOccup': float(input("Prosječna popunjenost: ")),
            'Latitude': float(input("Geografska širina: ")),
            'Longitude': float(input("Geografska dužina: "))
        }
        return features
    except ValueError:
        print("Greška: Molimo unesite ispravne brojčane vrijednosti.")
        return None

def main():
    """Glavna funkcija za interakciju s korisnikom"""
    print("\n=== PROCJENITELJ VRIJEDNOSTI NEKRETNINE ===")
    print("Dobrodošli u sustav za procjenu vrijednosti nekretnina!")
    
    model, scaler = load_model()
    if model is None or scaler is None:
        return
    
    while True:
        choice = input("\nŽelite li procjeniti vrijednost nekretnine? (da/ne): ").lower()
        
        if choice == 'ne':
            print("\nHvala na korištenju! Doviđenja!")
            break
        elif choice == 'da':
            features = get_user_input()
            if features is None:
                continue
                
            try:
                actual_price = float(input("\nUnesite stvarnu cijenu nekretnine (u 100000$): "))
                
                predicted_price = predict_house_price(model, scaler, features)
                if predicted_price is None:
                    continue
                
                print("\n=== REZULTATI PROCJENE ===")
                print(f"Predviđena cijena: {format_price(predicted_price)}")
                print(f"Stvarna cijena: {format_price(actual_price)}")
                print(evaluate_price(predicted_price, actual_price))
                
            except ValueError:
                print("Greška: Molimo unesite ispravnu cijenu.")
        else:
            print("Molimo odgovorite sa 'da' ili 'ne'")

if __name__ == "__main__":
    main()
