# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:28:33 2020
@author: utkuk
"""
# Gerekli paketler
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import imageio

img = imageio.imread("/Users/utkuk/Downloads/FonsuzTurkce/UygulamaDersleri/06GudumsuzOgrenme/sample.jpg")

w, h, d = orijinal_boyut = tuple(img.shape)

image_array = np.reshape(img, (w*h, d))

def imaj_yarat(kumeMerkezleri, etiket, w, h):
    """
    Kümelediğimiz noktaların küme merkezlerini içeren 360000x3'lük matrisi 600x600x3 boyuna çeviren fonksiyon.
    
    kumeMerkezleri: gruplandırma algoritması sonucu elde edilen kümelerin merkezleri
    
    etiket: veri içerisindeki noktaların model tarafından tahmin edilen etiketleri
    
    w: yaratılacak görüntünün genişliği
    
    h: yaratılacak görüntünün boyu
    """
    d = kumeMerkezleri.shape[1]
    gorsel = np.zeros((w, h, d))
    etiket_indis = 0
    for i in range(w):
        for j in range(h):
            gorsel[i][j] = kumeMerkezleri[int(etiket[int(etiket_indis)])]
            etiket_indis += 1
    return gorsel

# K-ortalamalar algoritma objesinin yaratılması, uydurulması ve modelin tahminlerinin kaydedilmesi
kOrtalama = KMeans(n_clusters = 10)
kOrtalama.fit(image_array)
etiketler = kOrtalama.predict(image_array)

Image.fromarray(imaj_yarat(kOrtalama.cluster_centers_, 
                           etiketler, w ,h).astype('uint8')) # imaj_yarat fonksiyonundan çıkan üç boyutlu matris ile görüntü oluşturma
    
# Hiyerarşik gruplandırma
from sklearn.cluster import AgglomerativeClustering
'''
Hiyerarşik sınıflandırmada her nokta arası uzaklığın bilinmesi gerekiyor. 
Fakat elimizdeki veri uzaklık matrisini oluştumak için çok büyük. 
Dolayısıyla sezgisel bir yöntem izleyerek ilk önce K-ortalamalar algoritmasıyla veri sayısını düşüreceğiz.
K-ortalamalar algoritmasından elde ettiğimiz merkezleri, hiyerarşik gruplandırma fonksiyonuna girdi olarak vereceğiz.
'''
# Baslangic k ortalamalar objesi, uydurumu, merkezlerin kaydedilmesi ve etiketlerin alınması
kOrtalamaHiyer = KMeans(n_clusters = 50,
                        random_state = 3)
kOrtalamaHiyer.fit(image_array)
kOrtalamaMerkezler = kOrtalamaHiyer.cluster_centers_
kOrtalamaTahmin = kOrtalamaHiyer.predict(image_array)

# Hiyerarşik gruplandırma objesinin belirlenen merkez sayısına göre yaratılması, uydurulması ve etiketlerin alınması
hiyerarsikKumeMerkezSayisi = 2
hiyerarsik = AgglomerativeClustering(n_clusters = hiyerarsikKumeMerkezSayisi, 
                                     linkage = 'single')
hiyerarsik.fit(kOrtalamaMerkezler)
hiyerarsik_etiketler = hiyerarsik.fit_predict(kOrtalamaMerkezler)    

'''
Elimizde 360000 tane nokta vardı. K-ortalamalar algoritmasıyla 50 merkez nokta belirledik çünkü 
50 merkez noktanın elimizdeki tüm veriyi aşağı yukarı temsil edeceğini düşündük. Daha fazla merkez nokta ile de bunu yapabilirdik.
Belirlediğimiz 50 merkez nokta ile hiyerarşik gruplandırma yaptık ve 2 tane kümeye ayırdık.
Şimdi bu 2 kümenin içerisinde bulunan toplam 50 merkezin içerisinde bulunan (K-ortalamalar ile elde edilen) 360000 noktanın etiketlerini (0 ya da 1)
ve merkezlerini bulacağız. 
'''
# Etiketlerini bulduğumuz döngü
gercek_etiketler = np.zeros((w*h))
for j in range(hiyerarsikKumeMerkezSayisi):
    kOrtalamaBagli = np.where(hiyerarsik_etiketler == j)[0]
    for k in range(len(kOrtalamaBagli)):
        gercek_etiketler[np.where(kOrtalamaTahmin == kOrtalamaBagli[k])] = j
# Merkezlerini bulduğumuz döngü
merkezler = np.zeros((hiyerarsikKumeMerkezSayisi, 3))
for i in range(hiyerarsikKumeMerkezSayisi):
    merkezler[i,:] = np.mean(image_array[np.where(gercek_etiketler == i)], 
                             axis = 0)
# Bulduğumuz etiket ve merkez verileriyle hiyeraşik gruplandırmanın görsel hali
Image.fromarray(imaj_yarat(merkezler, 
                           gercek_etiketler, w, h).astype('uint8'))
    
# Hiyerarşik gruplandırma, 5 küme merkezi ve tam bağlanım yöntemi ile
hiyerarsikKumeMerkezSayisi = 5
hiyerarsik = AgglomerativeClustering(n_clusters = hiyerarsikKumeMerkezSayisi, 
                                     linkage = 'complete')
hiyerarsik.fit(kOrtalamaMerkezler)
hiyerarsik_etiketler = hiyerarsik.fit_predict(kOrtalamaMerkezler)    

# Etiketlerini bulduğumuz döngü
gercek_etiketler = np.zeros((w*h))
for j in range(hiyerarsikKumeMerkezSayisi):
    kOrtalamaBagli = np.where(hiyerarsik_etiketler == j)[0]
    for k in range(len(kOrtalamaBagli)):
        gercek_etiketler[np.where(kOrtalamaTahmin == kOrtalamaBagli[k])] = j
# Merkezlerini bulduğumuz döngü
merkezler = np.zeros((hiyerarsikKumeMerkezSayisi, 3))
for i in range(hiyerarsikKumeMerkezSayisi):
    merkezler[i,:] = np.mean(image_array[np.where(gercek_etiketler == i)], 
                             axis = 0)
# Bulduğumuz etiket ve merkez verileriyle hiyeraşik gruplandırmanın görsel hali
Image.fromarray(imaj_yarat(merkezler, gercek_etiketler, w, h).astype('uint8'))
    
# Spektral Gruplandırma
from sklearn.cluster import SpectralClustering
# İstenen küme sayısı
spektralKumeSayisi = 5
# Spektral Gruplandırıcı objesinin yaratılması, veri kullanılarak uydurumu ve tahminlerin alınması
spektral = SpectralClustering(spektralKumeSayisi,
                              assign_labels = 'discretize',
                              random_state = 3)
spektral.fit(kOrtalamaMerkezler)
spektralTahminler = spektral.labels_

# Hiyerarşik gruplandırmadaki gibi 360000 veri noktasının etiketlerinin son modeldeki (spektral gruplandırma sonucunda elde edilen) verilerden elde edildiği döngüler
# Etiketlerini bulduğumuz döngü
gercek_etiketler_spektral = np.zeros((w*h))
for j in range(len(np.unique(spektralTahminler))):
    kOrtalamaBagli = np.where(spektralTahminler == j)[0]
    for k in range(len(kOrtalamaBagli)):
        gercek_etiketler_spektral[np.where(kOrtalamaTahmin == kOrtalamaBagli[k])] = j
# Merkezlerini bulduğumuz döngü
merkezler_spektral = np.zeros((len(np.unique(spektralTahminler)), 3))
for i in range(len(np.unique(spektralTahminler))):
    merkezler_spektral[i,:] = np.mean(image_array[np.where(gercek_etiketler_spektral == i)], 
                                      axis = 0)
# Bulduğumuz etiket ve merkez verileriyle hiyeraşik gruplandırmanın görsel hali
Image.fromarray(imaj_yarat(merkezler_spektral, 
                           gercek_etiketler_spektral, w, h).astype('uint8'))