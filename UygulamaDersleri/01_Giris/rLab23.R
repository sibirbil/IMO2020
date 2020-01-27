
# Fonksiyon çalistirma
## funcname(input1, input2)

# Fonksiyon yazma
carpim <- function(girdi1, girdi2){
  return(girdi1 * girdi2)
}
carpim(2, 5)

# Vektör yaratma
x = c(1,6,2)
x
y = c(1,4,3)
# Fonksiyon hakkinda yardim alma
?c
?length

# x+y, öncelikle x ve y'nin uzunluklarini kontrol ederek
length(x)
length(y)
x+y

# Yaratilan tüm objelerin listesi
ls()
# Obje silme
rm(x)
ls()
rm(list=ls()) # bütün hepsini silme

# Matris yaratimi
?matrix
x = matrix(data=c(1,2,3,4), nrow = 2, ncol = 2)
matrix(data=c(1,2,3,4), nrow = 2, ncol = 2, byrow = TRUE)

# Basit matris islemleri
x^2
x*2
x+2
sqrt(x)
x^(1/3)

# Normal dagilima göre dagilmis rassal degiskenler üretme
x = rnorm(50)
y = x + rnorm(50, mean = 50, sd = 0.1)
cor(x,y) # iki vektör arasindaki korelasyon katsayisi

mean(x) # ortalama
sd(x) # standart sapma

set.seed(1303) # tohum
rnorm(5)

# Grafik
x = rnorm(100)
y = rnorm(100)
?plot
plot(x,y)
plot(x,y, 
     type = 'p',
     xlab = 'this is the x-axis',
     ylab = 'this is the y-axis',
     main = 'X vs. Y')

# Sirali vektör yaratma
x = seq(1, 10)
x = seq(-pi, pi, length = 50)

y = x
f = outer(x,y,function(x,y)cos(y)/(1+x^2))
contour(x,y,f)
contour(x,y,f, nlevels = 45, add = T)
fa = (f-t(f))/2
contour(x,y,fa,nlevels = 45)
image(x,y,fa)
persp(x,y,fa)
persp(x,y,fa,theta=30)
persp(x,y,fa,theta=30,phi=50)

# Matris islemleri
A=matrix(1:16,4,4)
A

A[2,3]

A[c(1,3),c(2,4)]
A[1:3,2:4]
A[1:2,]
A[,1:2]
A[-c(1,3),]
dim(A)

# Veri okuma 
dogalgaz = read.table('C:/Users/utkuk/Downloads/dtist.csv', header = TRUE, sep = ',')
fix(dogalgaz) # Verileri elle düzeltme
names(dogalgaz) = c('ilce', 2015, 2016,2017,2018,2019) # Sütun isimlerini degistirme
dim(dogalgaz)

plot(dogalgaz$`2015`, dogalgaz$`2016`)
abline(coef = c(0,1))
text(dogalgaz$`2015`, dogalgaz$`2016`, dogalgaz$ilce, cex=0.6, pos=4, col="red") 

attach(dogalgaz)
plot(`2018`, `2019`)
abline(coef = c(0,1))
text(`2018`, `2019`, ilce, cex=0.6, pos=4, col="red")
hist(`2019`)
hist(`2019`, col = 2, breaks = 4)

summary(dogalgaz)

# Veri kaydetme
write.csv(dogalgaz, file = 'yeni_veri.csv')
write.table(mtcars, file = "yeni_veri.txt", sep = "\t",
            col.names = TRUE)

