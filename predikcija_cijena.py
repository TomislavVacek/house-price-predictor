import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje California Housing skupa podataka
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target  # Dodavanje ciljne varijable

# Ispis prvih pet redova i informacija o skupu podataka
print(data.head())
print(data.info())

# Definiranje X i y *prije* generiranja polinomnih značajki
X = data.drop('MedHouseVal', axis=1)  # Značajke
y = data['MedHouseVal']  # Ciljna varijabla (median house value)

# Vizualizacija i analiza podataka (opcionalno, ali korisno)
# Distribucija cijena
plt.figure(figsize=(8, 6))
sns.histplot(y, kde=True) # Ispravljeno: y umjesto data['MedHouseVal']
plt.title('Distribucija cijena kuća')
plt.show()

# Matrica korelacije
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matrica korelacije')
plt.show()

# Kreiranje polinomnih značajki
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Pretvaranje X_poly natrag u DataFrame s novim nazivima stupaca
feature_names = poly.get_feature_names_out(X.columns)
X_poly = pd.DataFrame(X_poly, columns=feature_names)

# Sada koristi X_poly umjesto X za podjelu podataka
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ispis prvih pet redova transformiranih podataka
print(X_poly.head())

# Treniranje modela linearne regresije
model = LinearRegression()
model.fit(X_train, y_train)

# Predviđanje na skupu za testiranje
y_pred = model.predict(X_test)

# Evaluacija modela
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-kvadrat: {r2}")
print(f"Srednja apsolutna pogreška: {mae}")
print(f"Srednja kvadratna pogreška: {mse}")

# Vizualizacija predviđenih vs. stvarnih cijena
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Stvarne cijene')
plt.ylabel('Predviđene cijene')
plt.title('Stvarne vs. Predviđene cijene')
plt.show()