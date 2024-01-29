# Kütüphaneleri Yükleyelim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data okuma
data = pd.read_csv("hw_25000.csv")

# Veriyi düzenleme
boy_kilo = data[['Height', 'Weight']]

# LinearRegression modelini eğitme
regression = LinearRegression()
regression.fit(boy_kilo[['Height']], boy_kilo['Weight'])

# Tahminler yapma ve sonuçları yazdırma
boy_values = np.array([[60], [62], [64], [66], [68], [70]])
predictions = regression.predict(boy_values.reshape(-1, 1))
for boy, prediction in zip(boy_values.flatten(), predictions):
    print(f"Boy: {boy}, Tahmin: {prediction}")

# R2 skoru hesaplama
r2 = r2_score(boy_kilo['Weight'], regression.predict(boy_kilo[['Height']]))
print(f"R2 Score: {r2}")

# Grafik çizme
plt.scatter(boy_kilo['Height'], boy_kilo['Weight'])
x_range = np.arange(min(boy_kilo['Height']), max(boy_kilo['Height'])).reshape(-1, 1)
plt.plot(x_range, regression.predict(x_range), color="red")
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Simple Linear Regression Model")
plt.show()
