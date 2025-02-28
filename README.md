# Regresyon Modelleri README

Bu proje, çeşitli regresyon modellerini kullanarak bir regresyon problemi çözmeyi amaçlar. Projede kullanılan modeller şunlardır:

- Decision Tree Regressor
- Multiple Linear Regression
- Polynomial Regression
- Random Forest Regressor
- Simple Linear Regression

## Modellerin Açıklamaları

1. **Decision Tree Regressor:**
   - Decision Tree algoritması, ağaç yapısı kullanarak bir bağımlı değişkeni tahmin eder.
   - Proje içinde `DecisionTreeRegressor` sınıfı kullanılmıştır.
   - Model, position level'a göre salary'i tahmin eder.
   - Karar ağacının tahmin ettiği değerler görselleştirilmiştir.

2. **Multiple Linear Regression:**
   - Multiple Linear Regression, birden fazla bağımsız değişkenin kullanıldığı bir regresyon yöntemidir.
   - Proje içinde `LinearRegression` sınıfı kullanılmıştır.
   - Model, age ve BMI'ye göre healthcare expenses'i tahmin eder.
   - Tahmin edilen değerler ile gerçek değerler karşılaştırılmıştır.

3. **Polynomial Regression:**
   - Polynomial Regression, doğrusal olmayan ilişkilere uyan bir regresyon yöntemidir.
   - Proje içinde `PolynomialFeatures` ve `LinearRegression` sınıfları kullanılmıştır.
   - Model, doğrusal ve polinom regresyonu aynı veri seti üzerinde karşılaştırır.
   - Polinom regresyonun doğrusal olmayan ilişkileri yakalayabileceği gösterilmiştir.
   - Her iki model de görselleştirilmiştir.

4. **Random Forest Regressor:**
   - Random Forest, birden fazla karar ağacını bir araya getirerek daha güçlü bir model oluşturan bir yöntemdir.
   - Proje içinde `RandomForestRegressor` sınıfı kullanılmıştır.
   - Model, position level'a göre salary'i tahmin eder.
   - Tek bir karar ağacından daha iyi tahmin performansı göstermiştir.
   - Özellik önem analizleri yapılmıştır.

5. **Simple Linear Regression:**
   - Simple Linear Regression, yalnızca bir bağımsız değişkenin kullanıldığı temel bir regresyon yöntemidir.
   - Proje içinde `LinearRegression` sınıfı kullanılmıştır.
   - Model, height'a göre weight'i tahmin eder.
   - Modelin performansı R² ve RMSE ile değerlendirilmiştir.
   - İlişki bir saçılma grafiği ve regresyon çizgisi ile görselleştirilmiştir.

## Kod Yapısı

Her model uygulaması tutarlı bir yapıya sahiptir:

- Hata işleme ile veri yükleme
- Model eğitimi
- Model değerlendirme (R², RMSE)
- Tahmin işlevi
- Sonuçların görselleştirilmesi
- Tüm iş akışını çalıştırmak için ana işlev

## Kurulum

Bu modelleri çalıştırmak için gerekli bağımlılıkları yüklemeniz gerekir:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Kullanım

Her model bağımsız olarak çalıştırılabilir. Örneğin, basit lineer regresyon modelini çalıştırmak için:

```bash
python basitLineer.py
```

## Veri Setleri

Modeller aşağıdaki veri setlerini kullanır:

- `hw_25000.csv`: Basit lineer regresyon için height ve weight verileri
- `insurance.csv`: Çoklu lineer regresyon için age, BMI ve healthcare expenses verileri
- `positions.csv`: Polinom, karar ağacı ve rastgele orman regresyonu için position level ve salary verileri

## Önemli Özellikler

- **Modüler Kod**: Her uygulama daha iyi okunabilirlik ve yeniden kullanılabilirlik için fonksiyonlara ayrılmıştır
- **Hata İşleme**: Dosya işlemleri ve veri işleme için uygun hata işleme
- **Belgeleme**: Tüm fonksiyonlar için kapsamlı docstring'ler
- **Görselleştirme**: Model performansı için net görselleştirmeler
- **Değerlendirme Ölçütleri**: Nicel model değerlendirme için R² puanı ve RMSE

## Gelecekteki İyileştirmeler

Bu projenin potansiyel iyileştirmeleri:

- Daha sağlam model değerlendirme için çapraz doğrulama ekleme
- Hiperparametre ayarlamasını uygulama
- Daha gelişmiş regresyon modellerini ekleme (örneğin, Destek Vektör Regresyonu, Gradient Boosting)
- Aynı veri setinde birden fazla modeli karşılaştırmak için birleşik bir arayüz oluşturma
