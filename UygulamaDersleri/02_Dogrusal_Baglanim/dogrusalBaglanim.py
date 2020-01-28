# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:35:50 2020
@author: utkuk
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tools # kaynak: https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/

auto = pd.read_csv('Auto.csv', index_col='name') # Dosyadan veri okuma, satır isimlerini 'name' adlı sütundan alarak
auto['ulke'] = auto.origin.replace([1,2,3],['USA','Europe','Japan']) # 'Origin' sütunundaki verilere göre yeni bir sütun yaratma
print(auto.head())
print(auto.index)
print(auto.columns)

auto.shape
auto.isnull().any() # auto verisinde boş bir değer var mı?
auto.dtypes # auto verisindeki sütunların veri tipleri
auto.horsepower.unique() # beygir gücü sütunundaki değerler
auto = auto[auto.horsepower != '?'] # beygir gücü sütununda soru işareti olmayan satırları seçme
print('?' in auto.horsepower) # soru işareti olan bir satır kaldı mı?
auto.shape 

auto.horsepower = auto.horsepower.astype('float') # beygir gücü sütununun veri tipini değiştirme
auto.dtypes
auto.describe() # auto verisinin kısa bir özeti

sns.distplot(auto['mpg']) # 'mpg' sütunundaki verilerin dağılımı

# Grafikler
fig = sns.boxplot(x='ulke', y='mpg', data=auto) # kutu grafiği çizdirme
plt.axhline(auto.mpg.mean(),color='r',linestyle='dashed',linewidth=2) # ortalama çizgisinin eklenmesi

fig = sns.boxplot(x='year', y='mpg', data=auto) # kutu grafiği çizdirme
plt.axhline(auto.mpg.mean(),color='r',linestyle='dashed',linewidth=2) # ortalama çizgisinin eklenmesi

fig = sns.boxplot(x='cylinders', y='mpg', data=auto) # kutu grafiği çizdirme
plt.axhline(auto.mpg.mean(),color='r',linestyle='dashed',linewidth=2) # ortalama çizgisinin eklenmesi

# Korelasyon matrisinin görselleştirilmesi
korelasyonMatrisi = auto.corr()
sns.heatmap(korelasyonMatrisi)

# Bazı faktörleri içeren korelasyon matrisinin görselleştirilmesi
faktorler = ['cylinders','displacement','horsepower','acceleration','weight','mpg']
korelasyonMatrisi = auto[faktorler].corr()
sns.heatmap(korelasyonMatrisi)

# A şıkkı
X = pd.DataFrame(auto['horsepower']) # girdi matrisinin sütunları
X = sm.add_constant(X) # beta_0 için 1'lerin eklenmesi
y = auto['mpg']  # bağımlı değişkenin tanımnlanması
lineerModel = sm.regression.linear_model.OLS(y, X) # model objesinin yaratılması
fit = lineerModel.fit() # modelin uydurulması
fit.summary() # modelin istatistiklerinin ve ayrıntılarının sunulduğu özeti

# A Şıkkı (iv)
predNew = fit.get_prediction([1, 98]) # modele dayalı beygir gücü 98 olan bir arabanın 'mpg' değeri
predNew.summary_frame(alpha=0.05) # %95 güven ve tahmin aralığı

# B şıkkı
plt.plot(X['horsepower'], y, 'o')
plt.plot(X['horsepower'], fit.fittedvalues, '-', lw=2)
plt.xlabel('beygir gücü')
plt.ylabel('uydurulmuş değerler (fitted values)')
plt.show()

# C şıkkı
tools.plots(fit, 'mpg', auto)

## 9. Soru
# A Şıkkı
pd.plotting.scatter_matrix(auto) # sütunların birbirine karşılık çizdirilmesi

# B Şıkkı
auto.corr() # sütunlar arası korelasyonu katsayılarını gösteren matris

# C Şıkkı
X = pd.DataFrame(auto.iloc[:,1:8]) # girdi matrisinin sütunları
X = sm.add_constant(X) # beta_0 için 1'lerin eklenmesi
y = auto['mpg'] # bağımlı değişkenin tanımnlanması
cokluDogrusalModel = sm.regression.linear_model.OLS(y, X) # model objesinin yaratılması
cokluDogrusalModelFit = cokluDogrusalModel.fit() # modelin uydurulması
cokluDogrusalModelFit.summary() # modelin istatistiklerinin ve ayrıntılarının sunulduğu özeti

# D şıkkı
tools.plots(cokluDogrusalModelFit, 'mpg', auto) # modeli incelemek için üretilen grafikler

# E şıkkı
import statsmodels.formula.api as smf # formulle doğrusal bağlanım yaratmak için gerekli kütüphane
model = smf.ols(formula='mpg ~ horsepower + weight + horsepower * weight + C(cylinders)', data=auto) # formül kullanarak yaratılan doğrusal bağlanım objesi
sonModel = model.fit() # modelin uydurulması
sonModel.summary() # modelin istatistiklerinin ve ayrıntılarının sunulduğu özeti
tools.residualPlot(sonModel, 'mpg', auto) # kalıntı grafiği

# Part F
auto['horsepower2'] = auto['horsepower']**2 # beygir gücünün karesinden yeni bir sütun yaratma
X = pd.DataFrame(auto.loc[:,['horsepower', 'horsepower2']]) # girdi matrisinin hazırlanması
X = sm.add_constant(X) # beta_0 için 1'lerin eklenmesi
y = auto['mpg'] # bağımlı değişken vektörünün tanımlanması
kareselModel = sm.regression.linear_model.OLS(y, X) # model objesinin yaratılması
kareselModelFit = kareselModel.fit() # modelin uydurulması
kareselModelFit.summary() # modelin istatistiklerinin ve ayrıntılarının sunulduğu özeti