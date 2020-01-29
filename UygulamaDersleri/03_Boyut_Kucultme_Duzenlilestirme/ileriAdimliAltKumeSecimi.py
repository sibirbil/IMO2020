# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:23:57 2020
@author: utkuk
"""
# Gerekli paketler
from sklearn.datasets import load_boston
import statsmodels.api as sm
import numpy as np
# Verinin yüklenmesi ve gerekli matrislerin hazırlanması
boston = load_boston()
X = boston.data
y = boston.target
X = sm.add_constant(X)
ozNitelikler = list(boston.feature_names)
ozNitelikler.insert(0, 'Intercept')
# Geri adımlı alt küme seçimi
for i in range(X.shape[1], 0, -1):
    lineerModel = sm.regression.linear_model.OLS(y, X) # doğrusal model objesinin yaratılması
    fit = lineerModel.fit() # modelin uydurulması
    pDegerleri = fit.pvalues # p değerleri
    maxYeri = np.where(pDegerleri == max(pDegerleri))[0][0] # en yüksek p değerine sahip özniteliğin bulunması
    print('Çıkarılacak öznitelik:', 
          ozNitelikler[maxYeri], 
          ' p-değeri', 
          pDegerleri[maxYeri])
    X = np.delete(X, maxYeri, axis = 1) # seçilen özniteliğin bilgilerinin modelden çıkarılması
    ozNitelikler.remove(ozNitelikler[maxYeri]) # çıkarılan özniteliğin listeden isminin silinmesi

