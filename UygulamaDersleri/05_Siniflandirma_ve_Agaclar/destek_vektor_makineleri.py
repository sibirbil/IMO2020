# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:02:46 2020
@author: utkuk
"""
# Gerekli paketler
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Verinin yüklenmesi
numaralar = load_digits()
X = numaralar.data
y = numaralar.target

# Ornek gorsel
fig = plt.figure()
plt.imshow(numaralar.images[0], cmap = plt.cm.binary)

# Model olusturma
dvm = SVC(C = 1, kernel = 'linear')

# Egitim, test verisi yaratma
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y,
                                                      test_size = 0.2,
                                                      shuffle = False,
                                                      random_state = 3)
# Modelin uydurulmasi
dvm.fit(X_egitim, y_egitim)

# Test verisinin tahminleri
tahmin_test = dvm.predict(X_test)

print('dvm için sınıflandırma raporu:\n', 
      classification_report(y_test, tahmin_test))

gorsel = plot_confusion_matrix(dvm, 
                               X_test,
                               y_test)
gorsel.figure_.suptitle("Hata matrisi")

print('Hata matrisi:\n', gorsel.confusion_matrix)




