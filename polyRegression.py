# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("positions.csv")

# Giriş ve çıkış verilerini düzenleme
level = data.iloc[:, 1].values.reshape(-1, 1)
salary = data.iloc[:, 2].values.reshape(-1, 1)

# Lineer Regresyon modeli eğitme
regression = LinearRegression()
regression.fit(level, salary)

# Lineer Regresyon ile tahmin yapma
tahmin = regression.predict(np.array([[8.3]]))

# Polinom Regresyon için veriyi dönüştürme
regressionPoly = PolynomialFeatures(degree=4)
levelPoly = regressionPoly.fit_transform(level)

# Polinom Regresyon modeli eğitme
regression2 = LinearRegression()
regression2.fit(levelPoly, salary)

# Polinom Regresyon ile tahmin yapma
tahmin2 = regression2.predict(regressionPoly.transform(np.array([[8.3]])))

# Grafik çizme
plt.scatter(level, salary, color="red", label="Gerçek Veri")
plt.plot(level, regression.predict(level), color="blue", label="Lineer Regresyon")
plt.plot(level, regression2.predict(levelPoly), color="green", label="Polinom Regresyon")
plt.title("Lineer ve Polinom Regresyon")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()
