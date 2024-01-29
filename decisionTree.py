# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Veriyi okuma
data = pd.read_csv("positions.csv")

# Giriş (level) ve çıkış (salary) verilerini düzenleme
level = data.iloc[:, 1].values.reshape(-1, 1)
salary = data.iloc[:, 2].values.reshape(-1, 1)

# Decision Tree Regressor modelini oluşturma ve eğitme
regression = DecisionTreeRegressor()
regression.fit(level, salary)

# Belirli bir level için tahmin yapma
prediction = regression.predict(np.array([[8.9]]))
print(f"Tahmin: {prediction[0]}")

# Grafik çizme
plt.scatter(level, salary, color="red", label="Gerçek Veri")
x_values = np.arange(min(level), max(level), 0.01).reshape(-1, 1)
plt.plot(x_values, regression.predict(x_values), color="orange", label="Decision Tree Tahmin")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Decision Tree Model")
plt.legend()
plt.show()
