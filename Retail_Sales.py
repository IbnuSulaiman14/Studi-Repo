import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3



conn = sqlite3.connect('techbox.db')
query = "SELECT * FROM penjualan WHERE cabang IN ('Jakarta','Bandung','Surabaya')"
df = pd.read_sql(query, conn)
print(df)

df['jumlah'] = df['jumlah'].fillna(1)

rata_rata = df['harga'].median()
rata_bulat = round(rata_rata)

df['harga'] = df['harga'].fillna(rata_bulat)

df.loc[df['jumlah'] == 1000, 'jumlah'] = 10

df['Total_Omzet'] = df['harga'] * df['jumlah']

print(df)

df['tanggal'] = pd.to_datetime(df['tanggal'])

plt.figure(figsize=(10,6))

sns.barplot(data=df, x='cabang', y='Total_Omzet', estimator=sum)

plt.title("Cabang dengan omzet terbanyak")
plt.xlabel("Cabang")
plt.ylabel("Total_Omzet")
plt.show()

plt.figure(figsize=(10,6))

sns.barplot(data=df, x='produk', y='jumlah')

plt.title("Produk Terlaris")
plt.xlabel("Produk")
plt.ylabel("Jumlah")
plt.show()

