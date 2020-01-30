# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:23:07 2020

@author: utkuk
"""
# Gerekli paketler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Verinin yüklenmesi
X, y = load_iris(return_X_y=True)

# Ayırma (Holdout)
# Eğitim ve test verilerinin oluşturulması
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 33)
# Verinin standartlaştırılması
X_egitim_standart = StandardScaler().fit_transform(X_egitim)
X_test_standart = StandardScaler().fit_transform(X_test)

keyk = KNeighborsClassifier(n_neighbors = 5,
                                      weights = 'uniform',
                                      p = 2,
                                      metric = 'minkowski',
                                      n_jobs = 1) # keyk model objesinin yaratılması
keyk.fit(X_egitim_standart, y_egitim) # keyk modelinin uydurulması
keyk_tahminler = keyk.predict(X_test_standart) # keyk modelinin test verisi için olan tahminleri
keyk_performans = keyk.score(X_test_standart, y_test) # keyk'in performans değeri

# Tekrarlamalı Ayırma (Repeated holdout)
t_keyk = KNeighborsClassifier(n_neighbors = 5,
                              weights = 'uniform',
                              p = 2,
                              metric = 'minkowski',
                              n_jobs = 1)
tekrar_sayisi = 50
dogruluk = []
for i in range(tekrar_sayisi):
    X_egitim, X_test, y_egitim, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = i)
    
    X_egitim_standart = StandardScaler().fit_transform(X_egitim)
    X_test_standart = StandardScaler().fit_transform(X_test)
    
    t_keyk.fit(X_egitim, y_egitim)    
    t_keyk_dogruluk = t_keyk.score(X_test, y_test)
    dogruluk.append(t_keyk_dogruluk)

dogruluk = np.asarray(dogruluk)
print('Tekrarlamalı ayırma performansı: ', dogruluk.mean())
t_keyk.fit(X,y)

# k-katlı çapraz geçerlilik sınaması (k-fold cross-validation)
hiperparametreler = range(1, 20)

# Eğitim ve test verilerinin oluşturulması
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 3)
# Verinin standartlaştırılması
X_egitim_standart = StandardScaler().fit_transform(X_egitim)
X_test_standart = StandardScaler().fit_transform(X_test)

# Çapraz geçerlilik sınaması (cgs) objesinin oluşturulması
cv = KFold(n_splits = 10, 
           shuffle = True)

# Model bilgilerinin tutulacağı vektörlerin oluşturulması
cgs_dogruluk, cgs_standartSap, cgs_standartHata = [], [], []

for k in hiperparametreler:
    k_cgs_keyk = KNeighborsClassifier(n_neighbors = k,
                                      weights = 'uniform',
                                      p = 2,
                                      metric = 'minkowski',
                                      n_jobs = 1)
    karesel_ortalama_hata = []
    for egitim_indisler, dogrulama_indisler in cv.split(X_egitim, y_egitim):
        tahminler = k_cgs_keyk.fit(X_egitim[egitim_indisler], 
                                    y_egitim[egitim_indisler]).predict(X_egitim[dogrulama_indisler])
        
        karesel_ortalama_hata_cgs = np.sqrt(np.mean(np.square(y_egitim[dogrulama_indisler] - tahminler)))
        karesel_ortalama_hata.append(karesel_ortalama_hata_cgs)
    
    karesel_ortalama_hata = np.array(karesel_ortalama_hata)
    y_tahmin_cgs10_ortalama = karesel_ortalama_hata.mean()
    y_tahmin_cgs10_standartSap = karesel_ortalama_hata.std()
    y_tahmin_cgs10_standartHata = y_tahmin_cgs10_standartSap / np.sqrt(10)
    
    cgs_dogruluk.append(y_tahmin_cgs10_ortalama)
    cgs_standartSap.append(y_tahmin_cgs10_standartSap)
    cgs_standartHata.append(y_tahmin_cgs10_standartHata)

en_iyi_K = np.argmax(cgs_dogruluk) # en iyi k değerini bulma

# en iyi k değeri ile yeni modeli yaratma ve başta belirlenen standartlaştırılmış eğitim verisiyle uydurma
k_cgs_keyk = KNeighborsClassifier(n_neighbors = hiperparametreler[en_iyi_K],
                                       weights = 'uniform',
                                       p = 2,
                                       metric = 'minkowski',
                                       n_jobs = 1)
k_cgs_keyk.fit(X_egitim_standart, y_egitim) 
k_cgs_keyk_y_tahmin_dogruluk = k_cgs_keyk.score(X_test_standart, y_test)

# Bütün veriyi kullanarak modele son şeklini verme
X_standart = StandardScaler().fit_transform(X)
k_cgs_keyk.fit(X_standart, y)
print('Çapraz geçerlilik sınaması performansı: ', k_cgs_keyk.score(X_standart, y))
# tekrarlamalı k katlı çapraz geçerlilik sınaması (repeated k-fold cv)
hiperparametreler = range(1, 20)
hiperparametre_genel_dogruluk = []; 
cgs_dogruluk = []

X_egitim, X_test, y_egitim, y_test = train_test_split(X, y,
                                                    test_size = 0.2)
X_egitim_standart = StandardScaler().fit_transform(X_egitim)
X_test_standart = StandardScaler().fit_transform(X_test)

tekrar_sayisi = 5
for i in range(tekrar_sayisi):
    cv = KFold(n_splits = 10, shuffle = True)
    hiperparametre_dogruluk = []
    for c in hiperparametreler:
        t_k_cgs_keyk = KNeighborsClassifier(n_neighbors = c,
                                                 weights = 'uniform',
                                                 p = 2,
                                                 metric = 'minkowski',
                                                 n_jobs = 1)
        karesel_ortalama_hata = []
        for egitim_indisler, dogrulama_indisler in cv.split(X_egitim, y_egitim):
            tahmin = t_k_cgs_keyk.fit(X_egitim[egitim_indisler], 
                                         y_egitim[egitim_indisler]).predict(X_egitim[dogrulama_indisler])
            
            karesel_ortalama_hata_cgs = np.sqrt(np.mean(np.square(y_egitim[dogrulama_indisler] - tahmin)))
            karesel_ortalama_hata.append(karesel_ortalama_hata_cgs)
        
        karesel_ortalama_hata = np.array(karesel_ortalama_hata)
    hiperparametre_dogruluk.append(karesel_ortalama_hata.mean())
    hiperparametre_genel_dogruluk.append(hiperparametre_dogruluk)

en_iyi_K = np.argmax(np.mean(hiperparametre_genel_dogruluk))
# en iyi k değeri ile yeni modeli yaratma ve başta belirlenen standartlaştırılmış eğitim verisiyle uydurma
t_k_cgs_keyk = KNeighborsClassifier(n_neighbors = hiperparametreler[en_iyi_K],
                                         weights = 'uniform',
                                         p = 2,
                                         metric = 'minkowski',
                                         n_jobs = 1)

t_k_cgs_keyk.fit(X_egitim_standart, y_egitim) 

t_k_cgs_y_tahmin_dogruluk = t_k_cgs_keyk.score(X_test_standart, y_test)

# Bütün veriyi kullanarak modele son şeklini verme
X_standart = StandardScaler().fit_transform(X)
t_k_cgs_keyk.fit(X_standart, y)
print('Tekrarlı çapraz geçerlilik sınaması performansı: ', t_k_cgs_keyk.score(X_standart, y))

