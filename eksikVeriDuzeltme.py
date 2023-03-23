import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# bosluklşarı doldurmak için kullandık
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
veriler = pd.read_csv("eksikveriler.csv")
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

yas = veriler.iloc[:, 1:4].values
print(yas)

imputer = imputer.fit(yas[:, 1:4])  # fit ile ogretiyoruz
yas[:, 1:4] = imputer.transform(yas[:, 1:4])  # transform ile de uytguşlamasını
print(yas)

ulke = veriler.iloc[:, 0:1].values
print(ulke)

# Ktegorik degerlerin numeric degerlere donusturulmesi
le = preprocessing.LabelEncoder()
ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0])
print(ulke)

# kolon baslıklarını etiketlere tasır
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr", "tr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=yas, index=range(
    22), columns=["boy", "kilo", "yas"])
print(sonuc2)
cinsiyet = veriler.iloc[:, -1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])
print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

x_train, x_test, y_train, y_test = train_test_split(
    s, sonuc3, test_size=0.33, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)

print(X_test)
print(X_train)
