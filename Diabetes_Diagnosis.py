import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# DATASET MEDIS
data_medis = {
    'Glukosa': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125, 110, 168, 139, 189, 166],
    'Tekanan_Darah': [72, 66, 64, 60, 40, 74, 50, 0, 70, 96, 92, 74, 80, 60, 72], # Ada 0 (Error)
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0, 37.6, 38.0, 27.1, 30.1, 25.8], # Ada 0 (Error)
    'Umur': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54, 30, 34, 57, 59, 51],
    'Diabetes': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1] # 1=Positif, 0=Negatif
}

df = pd.DataFrame(data_medis)

df['Tekanan_Darah'] = df['Tekanan_Darah'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0,np.nan)

rata_tekanandarah = df['Tekanan_Darah'].mean()
rata_BMI = df['BMI'].mean()

df['Tekanan_Darah'] = df['Tekanan_Darah'].fillna(round(rata_tekanandarah))
df['BMI'] = df['BMI'].fillna(round(rata_BMI))

X = df[['Glukosa','Tekanan_Darah','BMI','Umur']]
y = df['Diabetes']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train,y_train)

tes = model.predict(x_test)
evaluasi = accuracy_score(y_test, tes)

print(f"DATA ASLI : \n{y_test.values}")
print(f"Prediksi AI : \n{tes}")
print(f"Skor akurasi : \n{round(evaluasi*100)}%")

pasien_baru = [[150,70,30,45]]

prediksi = model.predict(pasien_baru)

if prediksi == 1:
    print("Pasien menderita Diabetes!")

else:
    print("Pasien Negatif Diabetes")