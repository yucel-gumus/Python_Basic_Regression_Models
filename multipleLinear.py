# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Veriyi okuma
data = pd.read_csv("insurance.csv")

# Veri sütunlarını göster
print("Veri Sütunları:", data.columns)

## Y ekseni (bağımlı değişken)
expenses = data.expenses.values.reshape(-1, 1)
## X ekseni (bağımsız değişkenler)
age_bmis = data.iloc[:, [0, 2]].values

# Lineer Regresyon modelini eğitme
regression = LinearRegression()
regression.fit(age_bmis, expenses)

# Belirli değerler için tahminler yapma
# Tahmin yaparken yaş ve vücut kitle indeksi (bmi) değerlerini içeren bir dizi kullanılır.
predictions = regression.predict(np.array([[30, 20], [30, 21], [20, 22], [20, 23], [20, 24]]))

# Tahmin sonuçlarını ekrana yazdırma
print("Tahmin Sonuçları:")
for i, prediction in enumerate(predictions):
    print(f"Tahmin {i+1}: {prediction[0]:.2f} (Sağlık Gideri)")

# Grafik çizme (veriyi ve regresyon çizgisini gösterme)
plt.scatter(age_bmis[:, 0], expenses, color="red", label="Gerçek Veri")
plt.scatter(age_bmis[:, 0], regression.predict(age_bmis), color="blue", label="Tahmin")
plt.xlabel("Yaş")
plt.ylabel("Sağlık Giderleri")
plt.title("Lineer Regresyon Tahmini")
plt.legend()
plt.show()
