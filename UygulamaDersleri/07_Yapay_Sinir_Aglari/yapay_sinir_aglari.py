# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:16:28 2020
@author: utkuk
"""

''' 
UCI Machine Learning Library
Bu kod içerisinde kullandığımız veri seti "UCI Machine Learning" kütüphanesin-
den alınmıştır. Kütüphanedeki diğer veri setleri için aşağıdaki linke bakabi -
lirsiniz.
http://archive.ics.uci.edu/ml/datasets.php
'''
# Gerekli paketler
import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils.random import sample_without_replacement

# Klasor degisikligi
os.chdir('/UygulamaDersleri/07YapaySinirAglariDerinOgrenme')

# Veriyi dosyadan okuma
veri_egitim = pd.read_csv('ann-train.data', sep = ' ', 
                          header = None)
veri_test = pd.read_csv('ann-test.data', sep = ' ', 
                          header = None)

# İndirdiğimiz veri boşluklarla ayrılmış bir veri. Her satır sonunda fazladan
# boşluk olduğu için onları atmamız gerekiyor. Alttaki satırlar bu işlemi 
# gerçekleştiriyor.
veri_egitim.drop(veri_egitim.columns[[22,23]], axis = 1, 
                 inplace = True)
veri_test.drop(veri_test.columns[[22,23]], axis = 1, 
                 inplace = True)

X_egitim = veri_egitim.iloc[:, range(21)]
X_test = veri_test.iloc[:, range(21)]

y_egitim = veri_egitim.iloc[:, 21]
y_test = veri_test.iloc[:, 21]

# Siniflara ait ornek sayilari
print('1. sınıfa ait olan örnek sayısı:', sum(y_egitim == 1))
print('2. sınıfa ait olan örnek sayısı:', sum(y_egitim == 2))
print('3. sınıfa ait olan örnek sayısı:', sum(y_egitim == 3))
# Verimizde örneklerin ait olduğu sınıflarda bir dengesizlik mevcut.

# Yapay sinir ağı (ysa) oluşturma, eğitme ve sonuçlarını ekrana yazdırma.
ysa = MLPClassifier(hidden_layer_sizes = (5,),
                    activation = 'tanh',
                    solver = 'sgd',
                    max_iter = 5000,
                    random_state = 3)
ysa.fit(X_egitim, y_egitim)
print(ysa.score(X_test, y_test))
print(confusion_matrix(y_test, ysa.predict(X_test)))
print(classification_report(y_test, ysa.predict(X_test)))

# Gizli katmandaki düğüm sayısını bulmak için çapraz geçerlilik sınaması yapma
parametreler = range(2, 11)
cgs = KFold(n_splits = 10, shuffle = True, random_state = 4)
parametre_performans = []
for c in parametreler:
    cgs_performans = []
    for egitim_indis, dogrulama_indis in cgs.split(X_egitim, y_egitim):
        cgs_ysa = MLPClassifier(hidden_layer_sizes = (c, ),
                                activation = 'tanh',
                                solver = 'sgd',
                                max_iter = 500,
                                learning_rate_init = 0.01,
                                random_state = 5)
        cgs_ysa.fit(X_egitim.iloc[egitim_indis, :], 
                    y_egitim.iloc[egitim_indis])
        dogruluk = cgs_ysa.score(X_egitim.iloc[dogrulama_indis,:],
                                  y_egitim.iloc[dogrulama_indis])
        cgs_performans.append(dogruluk)
    cgs_performans = np.array(cgs_performans)
    parametre_performans.append(cgs_performans.mean())

# En iyi performansı gösteren düğüm/nöron sayısını bulma
en_iyi_noron_sayisi = parametreler[np.argmax(parametre_performans)]
# En iyi performansı gösteren düğüm/nöron sayısı ile yeni bir YSA modeli
# yaratma, eğitme ve performansını ekrana yazdırma.    
en_iyi_ysa = MLPClassifier(hidden_layer_sizes = en_iyi_noron_sayisi,
                           activation = 'tanh',
                           solver = 'sgd',
                           max_iter = 500,
                           learning_rate_init = 0.01,
                           random_state = 5)

en_iyi_ysa.fit(X_test, y_test)
dogruluk_test = en_iyi_ysa.score(X_test, y_test)
print('10 katlı CGS tahmin doğruluğu:', dogruluk_test)
print(confusion_matrix(y_test, en_iyi_ysa.predict(X_test)))
print(classification_report(y_test, en_iyi_ysa.predict(X_test)))

'''
Sınıflar arasındaki dengesizliği gidermek için örnek sayısı çok olan sınıf -
sayısı az olan sınıftaki örnek sayısı kadar örnek seçmek. Elimizdeki veriye
göre konuşacak olursak, 1. sınıfa ait 93 tane örneğimiz var. 2. ve 3. sınıf-
tan 93'er tane rastgele örnek seçeceğiz. Yaratacağımız modeli toplamda 279 
tane örnekle eğiteceğiz ve test verimizi bu modelle tahmin edeceğiz.
'''
# Sınıfların elimizdeki eğitim verisinde hangi satırlarda olduğunu kaydetme
indisler_1 = np.where(y_egitim == 1)
indisler_2 = np.where(y_egitim == 2)
indisler_3 = np.where(y_egitim == 3)

# Yukarıda belirlediğimiz satır sayıları arasından istenen sayıda (1. sınıfta
# bulunan örnek sayısı kadar sayıda) örnek seçme
ornekler2 = sample_without_replacement(n_population = sum(y_egitim == 2),
                                       n_samples = sum(y_egitim == 1))
ornekler3 = sample_without_replacement(n_population = sum(y_egitim == 3),
                                       n_samples = sum(y_egitim == 1))
# Yarattığımız alt kümelerden dengelenmiş (279 tane örnek içeren)
# veri seti yaratma
y_egitim_dengelenmis = y_egitim[indisler_1[0]]
y_egitim_dengelenmis = np.append(y_egitim_dengelenmis,
                                 y_egitim[indisler_2[0][ornekler2]])
y_egitim_dengelenmis = np.append(y_egitim_dengelenmis,
                                 y_egitim[indisler_3[0][ornekler3]])
dengelenmis_indisler = np.append(np.array(indisler_1), 
                                 np.array(indisler_2[0][ornekler2]))
dengelenmis_indisler = np.append(dengelenmis_indisler, 
                                 np.array(indisler_3[0][ornekler3]))
X_egitim_dengelenmis = X_egitim.iloc[dengelenmis_indisler,:]        

#Dengelenmiş set için model yaratma, eğitme ve performansını ekrana yazdırma
dengelenmis_ysa = MLPClassifier(hidden_layer_sizes = en_iyi_noron_sayisi,
                                activation = 'tanh',
                                solver = 'sgd',
                                learning_rate_init = 0.01,
                                max_iter = 500,
                                random_state = 5)
dengelenmis_ysa.fit(X_egitim_dengelenmis, y_egitim_dengelenmis)
dengelenmis_dogruluk = dengelenmis_ysa.score(X_test, y_test)
print('Dengelenmiş veri seti ile eğitilmiş modelin doğruluğu:', 
      dengelenmis_dogruluk)
print(confusion_matrix(y_test, 
                       dengelenmis_ysa.predict(X_test)))
print(classification_report(y_test, 
                            dengelenmis_ysa.predict(X_test)))
















