# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:22:33 2020
@author: utkuk
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import numpy as np

X, y = load_boston(return_X_y = True) # Boston veri setinin yüklenmesi

X = np.c_[np.ones([506, 1]), X] # beta_0 için 1'lerin eklenmesi

beta = np.linalg.multi_dot([np.linalg.inv(np.matmul(X.T, X)), X.T, y]) # (X^T * X)^-1 * X^T * y

reg = LinearRegression().fit(X, y) # doğrusal bağlanım objesinin oluşturulması ve girdilerle uydurulması
reg.coef_ # beta değerleri
reg.intercept_ # beta_0 değerleri