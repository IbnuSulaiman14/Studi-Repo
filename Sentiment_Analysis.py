import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



kalimat = [
    "Barang bagus banget", "Pengiriman cepat dan aman", "Suka sekali warnanya", 
    "Kualitas mantap harga murah", "Sangat memuaskan", "Packing rapih", 
    "Recommended seller", "Produk original", "Pelayanan ramah", "Bintang lima pokoknya",
    "Barang jelek parah", "Pengiriman lama banget", "Warna tidak sesuai foto",
    "Kualitas sampah", "Kecewa berat", "Packing hancur", 
    "Penipu barang tidak dikirim", "Rusak pas sampai", "Pelayanan kasar", "Jangan beli di sini"
]

# Label: 1 = Positif, 0 = Negatif
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(kalimat)
word_index = tokenizer.word_index

sequence = tokenizer.texts_to_sequences(kalimat)

padded = pad_sequences(sequence, maxlen=5,padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded,labels, test_size=0.2, random_state=42)

model = Sequential()

model.add(Dense(units=16, input_shape=(5,), activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,y_train, epochs=100)

acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(len(acc))


plt.figure(figsize=(8,4))
plt.plot(epochs_range,acc,label='Training Akurasi')
plt.plot(epochs_range, loss, label='Training Loss', linestyle='--')
plt.title("Sentiment Analysis")
plt.xlabel("Epochs")
plt.ylabel("Nilai")
plt.show()


print("="*50)
print("KOMENTAR SENTIMEN ANALISIS")
print("Ketik 'X' untuk keluar")
while True:
    komentar = input("Masukkan komentar anda : ")
    if komentar.lower() == 'x':break
    
    test_pad = tokenizer.texts_to_sequences([komentar])
    test_seq = pad_sequences(test_pad, maxlen=5,padding='post', truncating='post')
    print(f"Angka : {test_seq}")
    
    prediksi = model.predict(test_seq)
    
    hasil = prediksi[0][0]
    print(f"Hasil : {hasil}")
    if hasil > 0.5:
        print("Komentar ini positif")
        
    else:
        print("Komentar NEGATIF")
    