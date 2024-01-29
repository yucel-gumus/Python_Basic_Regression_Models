# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Veriyi yükleyen fonksiyon
def load_data(file_path):
    """
    CSV dosyasındaki veriyi yükler ve giriş ve çıkış değişkenlerini döndürür.

    Parameters:
    - file_path (str): Verinin bulunduğu CSV dosyasının yolu.

    Returns:
    - level (numpy.ndarray): İş seviyesi (giriş değişkeni).
    - salary (numpy.ndarray): Maaş (çıkış değişkeni).
    """
    data = pd.read_csv(file_path)
    level = data.iloc[:, 1].values.reshape(-1, 1)
    salary = data.iloc[:, 2].values
    return level, salary

# Random Forest Regressor modelini eğiten fonksiyon
def train_random_forest(level, salary, n_estimators=10, random_state=2):
    """
    Random Forest Regressor modelini eğitir.

    Parameters:
    - level (numpy.ndarray): İş seviyesi (giriş değişkeni).
    - salary (numpy.ndarray): Maaş (çıkış değişkeni).
    - n_estimators (int): Oluşturulacak ağaç sayısı.
    - random_state (int): Rastgele sayı üretimi için seed değeri.

    Returns:
    - regression_model (RandomForestRegressor): Eğitilmiş Random Forest modeli.
    """
    regression_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    regression_model.fit(level, salary)
    return regression_model

# Modeli kullanarak tahmin yapan fonksiyon
def make_prediction(model, value):
    """
    Verilen modeli kullanarak bir değeri tahmin eder.

    Parameters:
    - model (RandomForestRegressor): Tahmin yapmak için kullanılacak model.
    - value (float): Tahmin yapılacak değer.

    Returns:
    - prediction (numpy.ndarray): Tahmin sonucu.
    """
    prediction = model.predict(np.array([value]).reshape(-1, 1))
    return prediction

# Ana program
def main():
    # Veriyi yükleme
    file_path = "positions.csv"
    level, salary = load_data(file_path)
    
    # Random Forest modelini eğitme
    regression_model = train_random_forest(level, salary)
    
    # Tahmin yapma
    prediction_value = 8.3
    prediction = make_prediction(regression_model, prediction_value)
    
    # Tahmin sonucunu ekrana yazdırma
    print(prediction)

if __name__ == "__main__":
    main()
