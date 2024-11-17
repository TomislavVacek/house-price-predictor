import joblib
import pandas as pd
import numpy as np

def load_model():
    try:
        model = joblib.load('house_price_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except:
        print("Error: Model files not found!")
        return None, None

def predict_house_price(model, scaler, features):
    """
    Funkcija koja predviđa cijenu kuće na temelju unesenih karakteristika.
    """
    input_data = pd.DataFrame([features])
    
    input_data['rooms_per_household'] = input_data['AveRooms'] / input_data['AveOccup']
    input_data['bedrooms_per_room'] = input_data['AveBedrms'] / input_data['AveRooms']
    input_data['population_per_household'] = input_data['Population'] / input_data['AveOccup']
    input_data['location'] = input_data['Latitude'] * input_data['Longitude']
    
    for col in ['MedInc', 'Population', 'AveOccup']:
        input_data[f'log_{col}'] = np.log1p(input_data[col])
    
    input_scaled = scaler.transform(input_data)
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
        
        model, scaler = load_model()
        if model is None or scaler is None:
            return
        
        predicted_price = predict_house_price(model, scaler, features)
        
        print("\n=== REZULTATI PROCJENE ===")
        print(f"Predviđena cijena: ${predicted_price*100000:.2f}")
        print(f"Stvarna cijena: ${actual_price*100000:.2f}")
        print(evaluate_price(predicted_price, actual_price))
        
    except ValueError:
        print("Pogreška: Molimo unesite važeće brojčane vrijednosti.")
    except Exception as e:
        print(f"Došlo je do pogreške: {str(e)}")

if __name__ == "__main__":
    while True:
        choice = input("\nŽelite li predvidjeti cijenu nekretnine? (da/ne): ").lower()
        if choice == 'da':
            predict_interface()
        elif choice == 'ne':
            break
        else:
            print("Molimo odgovorite sa 'da' ili 'ne'")
