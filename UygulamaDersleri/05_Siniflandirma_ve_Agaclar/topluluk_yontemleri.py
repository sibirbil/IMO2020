# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:49:52 2020

@author: utkuk
"""
# Gerekli paketler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_wine
# Veriyi yükleme
wine = load_wine()
X = wine.data
y = wine.target

cgs = KFold(n_splits=10) # 10 katlı çapraz geçerlilik sınaması objesinin yaratılması
agac = DecisionTreeClassifier(criterion = 'entropy') # torbalamada ve AdaBoost'ta kullanılacak karar ağacı objesinin yaratılması

torbalama = BaggingClassifier(base_estimator = agac, 
                              n_estimators = 5) # agac objesinin kullanılarak torbalama modelinin kurulması
torbalama_performans = cross_val_score(torbalama, X, y, cv = cgs) # yaratılan 10 katlı çapraz geçerlilik sınaması ile torbalama modelinin performansının kaydedilmesi
print('Torbalama performansı:', torbalama_performans.mean())

rastgele_ormanlar = RandomForestClassifier(n_estimators = 5, 
                                           max_features = 3) # her ağaçta maksimum 3 öznitelik kullanılarak 5 farklı ağaç oluşturan rastgele ormanlar objesinin yaratılması
rastgele_ormanlar_performans = cross_val_score(rastgele_ormanlar, X, y, cv = cgs) # yaratılan 10 katlı çapraz geçerlilik sınaması ile rastgele ormanlar modelinin performansının kaydedilmesi
print('Rastgele ormanlar performansı:', rastgele_ormanlar_performans.mean())

adaboost = AdaBoostClassifier(base_estimator = agac, n_estimators=1000) # agac objesini kullanarak maksimüm 1000 döngü ile eğitilecek AdaBoost objesinin yaratılması
adaboost_performans = cross_val_score(adaboost, X, y, cv = cgs) # yaratılan 10 katlı çapraz geçerlilik sınaması ile AdaBoost modelinin performansının kaydedilmesi
print('Adaboost performansi:', adaboost_performans.mean())
