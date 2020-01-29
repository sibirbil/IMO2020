# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:58:46 2020

@author: utkuk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y = True)
boston = load_boston()
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y,
                                                      test_size = 0.3,
                                                      random_state = 3)
# Lineer Model
lineerModel = LinearRegression() # model objesinin yaratılması
lineerModel.fit(X_egitim, y_egitim) # modelin eğitim verisi kullanılarak uydurulması
lineer_egitim_r2 = lineerModel.score(X_egitim, y_egitim) # modelin eğitim verisi üzerinde R^2 değeri
lineer_test_r2 = lineerModel.score(X_test, y_test) # modelin test verisi üzerinde R^2 değeri
print('Lineer: Egitim verisi R2 degeri ', lineer_egitim_r2)
print('Lineer: Test verisi R2 degeri', lineer_test_r2)

# Ridge
ridge = Ridge(alpha = 0.01) # Ridge model objesinin yaratılması, lambda = 0.01 
ridge.fit(X_egitim, y_egitim) # modelin eğitim verisi kullanılarak uydurulması
ridge_egitim_r2 = ridge.score(X_egitim, y_egitim)
ridge_test_r2 = ridge.score(X_test, y_test)
print('Ridge: Egitim verisi R2 degeri', ridge_egitim_r2)
print('Ridge: Test verisi R2 degeri', ridge_test_r2)

ridge2 = Ridge(alpha = 100)
ridge2.fit(X_egitim, y_egitim)
ridge2_egitim_r2 = ridge2.score(X_egitim, y_egitim)
ridge2_test_r2 = ridge2.score(X_test, y_test)
print('Ridge2: Egitim verisi R2 degeri', ridge2_egitim_r2)
print('Ridge2: Test verisi R2 degeri', ridge2_test_r2)

# Grafikler
plt.figure(figsize=(8,6))
plt.plot(ridge.coef_, 
         alpha =0.7,
         linestyle = 'none',
         marker = '*',
         markersize = 15,
         color = 'red',
         label = r'Ridge: $\lambda = 0.01$')
plt.plot(ridge2.coef_,
         alpha = 0.5,
         linestyle = 'none',
         marker = 'd',
         markersize = 15,
         color = 'blue',
         label = r'Ridge: $\lambda = 100$')
plt.plot(lineerModel.coef_,
         alpha = 0.4,
         linestyle = 'none',
         marker = 'o',
         markersize = 15,
         color = 'green',
         label = 'Lineer Model')

plt.xlabel('Katsayılar', fontsize = 16)
plt.ylabel('Katsayı değerleri', fontsize = 16)
plt.legend(fontsize = 13, loc = 4)
plt.show()

# Lasso 
lasso = Lasso(alpha = 0.01) # Lasso model objesinin yaratılması, lambda = 0.01 
lasso.fit(X_egitim, y_egitim)
lasso_egitim_r2 = lasso.score(X_egitim, y_egitim)
lasso_test_r2 = lasso.score(X_test, y_test)
print('Lasso: Egitim verisi R^2 değeri', lasso_egitim_r2)
print('Lasso: Test verisi R^2 değeri', lasso_test_r2)

lasso2 = Lasso(alpha = 1)
lasso2.fit(X_egitim, y_egitim)
lasso2_egitim_r2 = lasso2.score(X_egitim, y_egitim)
lasso2_test_r2 = lasso2.score(X_test, y_test)
print('Lasso2: Egitim verisi R^2 değeri', lasso2_egitim_r2)
print('Lasso2: Test verisi R^2 değeri', lasso2_test_r2)

lasso3 = Lasso(alpha = 100)
lasso3.fit(X_egitim, y_egitim)
lasso3_egitim_r2 = lasso3.score(X_egitim, y_egitim)
lasso3_test_r2 = lasso3.score(X_test, y_test)
print('Lasso3: Egitim verisi R^2 değeri', lasso3_egitim_r2)
print('Lasso3: Test verisi R^2 değeri', lasso3_test_r2)

# Grafikler
plt.figure(figsize=(8,6))
plt.plot(lasso.coef_,
         alpha = 0.7,
         linestyle = 'none',
         marker = '*',
         markersize = 15,
         color = 'red',
         label = r'Lasso: $\lambda = 0.01$')
plt.plot(lasso2.coef_,
         alpha = 0.7,
         linestyle = 'none',
         marker = 'd',
         markersize = 15,
         color = 'blue',
         label = r'Lasso: $\lambda = 1$')
plt.plot(lasso3.coef_,
         alpha = 0.7,
         linestyle = 'none',
         marker = 's',
         markersize = 15,
         color = 'black',
         label = r'Lasso: $\lambda = 100$')
plt.plot(lineerModel.coef_,
         alpha = 0.7,
         linestyle = 'none',
         marker = 'v',
         markersize = 15,
         color = 'green',
         label = r'Lineer Model$')
plt.xlabel('Öznitelikler', fontsize = 20)
plt.ylabel('Öznitelik Katsayıları', fontsize = 20)
plt.legend(fontsize = 18, loc = 4)
plt.show()

# Öznitelik Katsayı incelemesi
lasso_katsayilar = sum(lasso.coef_ == 0)
lasso2_katsayilar = sum(lasso2.coef_ == 0)
lasso3_katsayilar = sum(lasso3.coef_ == 0)

print('Lasso (lambda = 0.01): ', lasso_katsayilar)
print('Lasso (lambda = 1): ', lasso2_katsayilar)
print('Lasso (lambda = 100): ', lasso3_katsayilar)

# Lasso2 modelinde beta katsayısı 0 olan özniteliklerin bulunması
print(boston.feature_names[np.where(lasso2.coef_ == 0)[0]])



