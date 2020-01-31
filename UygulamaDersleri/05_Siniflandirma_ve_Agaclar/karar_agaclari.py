# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:45:34 2020

@author: utkuk
"""
# Gerekli paketlerin çağrılması
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn import tree

# Verinin yüklenmesi, eğitim ve test verileri olarak ayrılması
boston=load_boston()
X = boston.data
Y = boston.target
X_egitim,X_test,y_egitim,y_test = train_test_split(X, Y, 
                                                 test_size = 0.3, 
                                                 random_state=3)

# Bağlanım (Regresyon) Karar Ağacı objesinin oluşturulması, eğitim verileriyle uydurulması, eğitim ve test verileri üzerinden performansının belirlenmesi 
regTree = DecisionTreeRegressor(criterion = "mse",
                                max_depth = None,
                                min_samples_leaf = 2,
                                random_state = 3)
regTree.fit(X_egitim, y_egitim)
regTree_train_score = regTree.score(X_egitim, y_egitim)
regTree_test_score = regTree.score(X_test, y_test)
print("1. Regresyon ağacı için eğitim performansı:", regTree_train_score)
print("1. Regresyon ağacı için test performansı: ", regTree_test_score)

fig = plt.figure(figsize = (20, 9))
tree.plot_tree(regTree)
tree.plot_tree(regTree, max_depth = 3, filled=True, proportion = True)
# Bağlanım (Regresyon) Karar Ağacı objesinin oluşturulması, eğitim verileriyle uydurulması, eğitim ve test verileri üzerinden performansının belirlenmesi 
regTree2 = DecisionTreeRegressor(criterion = "mse",
                                max_depth = None,
                                min_samples_leaf = 10,
                                random_state = 3)
regTree2.fit(X_egitim, y_egitim)
regTree2_train_score = regTree2.score(X_egitim, y_egitim)
regTree2_test_score = regTree2.score(X_test, y_test)
print("2. Regresyon ağacı için eğitim performansı:", regTree2_train_score)
print("2. Regresyon ağacı için test performansı:", regTree2_test_score)
# Bağlanım (Regresyon) Karar Ağacı objesinin oluşturulması, eğitim verileriyle uydurulması, eğitim ve test verileri üzerinden performansının belirlenmesi 
regTree3 = DecisionTreeRegressor(criterion = "mse",
                                max_depth = 3,
                                min_samples_leaf = 2,
                                random_state = 3)
regTree3.fit(X_egitim, y_egitim)
regTree3_train_score = regTree3.score(X_egitim, y_egitim)
regTree3_test_score = regTree3.score(X_test, y_test)
print("3. Regresyon ağacı için eğitim performansı:", regTree3_train_score)
print("3. Regresyon ağacı için test performansı: ", regTree3_test_score)

# alfa hiperparametresinin çapraz geçerlilik sınaması ile belirlenmesi
hiperparametreler = np.linspace(0, 3, 50)
cv = KFold(n_splits = 5,
           shuffle = True,
           random_state = 3)

hiperparametre_dogruluk = []
for c in hiperparametreler:
    cv_dogruluk = []
    for eğitim_indisleri, dogrulama_indisleri in cv.split(X_egitim, y_egitim):
        agac = DecisionTreeRegressor(criterion = "mse",
                                     ccp_alpha = c,
                                     random_state = 3)
        agac.fit(X_egitim[eğitim_indisleri], y_egitim[eğitim_indisleri])
        dogruluk = agac.score(X_egitim[dogrulama_indisleri], y_egitim[dogrulama_indisleri])
        cv_dogruluk.append(dogruluk)
    cv_dogruluk = np.array(cv_dogruluk)
    hiperparametre_dogruluk.append(cv_dogruluk.mean())

best_alpha = hiperparametreler[np.argmax(hiperparametre_dogruluk)]
agac = DecisionTreeRegressor(criterion = "mse",
                             ccp_alpha = best_alpha,
                             random_state = 3).fit(X_egitim, y_egitim)
dogruluk = agac.score(X_test, y_test)
print("5-katlı ÇGS sonucu seçilen alfa değeri ile elde edilen tahmin doğruluğu:", dogruluk)

# Yinelemeli Öznitelik Eleme - Geri adımlı alt küme seçimine benzer bir yaklaşım 
# Recursive feature elimination (Similar to backward stepwise)
tahminleyici = LinearRegression()
oznitelik_eleyici = RFE(tahminleyici, 
                        n_features_to_select = 7, 
                        step = 1) # 13'ten 7 tane özniteliğe indirme
oznitelik_eleyici.fit(X_egitim, y_egitim) # modele uydurma
boston.feature_names[oznitelik_eleyici.get_support()]
oznitelik_egitim_performans = oznitelik_eleyici.score(X_egitim, y_egitim)
oznitelik_test_performans = oznitelik_eleyici.score(X_test, y_test)
print("Öznitelik eleyici ile elde edilen modelin eğitim verisi üzerine tahminlerinin doğruluğu:", oznitelik_egitim_performans)
print("Öznitelik eleyici ile elde edilen modelin test verisi üzerine tahminlerinin doğruluğu:", oznitelik_test_performans)