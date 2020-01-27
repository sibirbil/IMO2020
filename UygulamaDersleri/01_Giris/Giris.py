# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:13:05 2020
@author: utkuk
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Vektör yaratma
x = np.array([1, 6, 2])
print(x)
y = np.array([1, 4, 3])
# Fonksiyon hakkında yardım
help(np.array)
help(len)
# x+y, öncelikle boyutları uyuyor mu diye kontrol ederek
len(x)
len(y)
x+y

# tanımladığımız değişkeni silme
del x

# Matris yaratımı
help(np.matrix)
x = np.array([1, 2, 3, 4])
x.shape = (2,2)
x = np.array([1, 2, 3, 4]).reshape(2,2)
# Matrislerde basit işlemler
x*x
x*2
x+2
np.sqrt(x)
np.power(x, 1/3)

# Normal dağılıma göre dağılmış rassal değişkenler yaratma
x = np.random.randn(50)
y = x + np.random.normal(loc = 50, 
                         scale = 0.1, 
                         size = 50)
np.corrcoef(x,y) # İki vektör arasındaki korelasyon katsayısı

np.mean(x) # Ortalama
np.std(x) # Standart sapma

np.random.seed(12) # Tohum
np.random.randn(5)

# Grafik çizme
x = np.random.randn(100)
y = np.random.randn(100)

plt.plot(x,y, 'bo')
plt.show()

plt.ylabel('this is the y-axis')
plt.xlabel('this is the x-axis')
plt.title('X vs Y')

# Sıralı vektör yaratma
x = np.arange(1, 10)
x = np.linspace(-np.pi, np.pi, num = 50)
xx = np.transpose(np.tile(x, (50,1)))

y = x
yy = np.tile(y, (50,1))
f = np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        f[i,j] = np.cos(y[j])/(1+x[i]**2)
plt.contour(f)
plt.contour(f, levels = 45)

fa = (f-np.transpose(f))/2
plt.contour(fa, levels = 45)
plt.imshow(fa)

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, fa, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(xx, yy, fa, zdir='fa', offset=-2, cmap=cm.coolwarm)
cset = ax.contourf(xx, yy, fa, zdir='x', offset=-5, cmap=cm.coolwarm)
cset = ax.contourf(xx, yy, fa, zdir='y', offset=-5, cmap=cm.coolwarm)

# Matris işlemleri
A=np.matrix([np.arange(1, 17)]).reshape(4, 4, order = 'F')
A

A[1,2]

A[np.ix_([0, 2],[1, 3])]
A[0:3,1:4]
A[0:2,:]
A[:,0:2]
np.delete(A, [0, 2], axis=0)
np.shape(A)

# Veri okuma
dogalgaz = pd.read_csv('dtist.csv', sep = ',')
dogalgaz = dogalgaz.rename(columns={"İlçe": "ilce"})
dogalgaz.columns
dogalgaz.shape

fig = plt.figure()
ax = fig.gca()
ax.plot(dogalgaz['2015'], dogalgaz['2016'], 'bo')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # iki eksenin minimumu
    np.max([ax.get_xlim(), ax.get_ylim()]),  # iki eksenin maksimumu
]
# limitlerini birbirine karşılık çizdirme
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
for i, txt in enumerate(dogalgaz['ilce']):
    ax.annotate(txt, (dogalgaz['2015'][i], dogalgaz['2016'][i]))
    
plt.hist(dogalgaz['2019'])
plt.hist(dogalgaz['2019'], bins = 4)

dogalgaz.info()
dogalgaz.describe()
