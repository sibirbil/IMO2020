
# Gerekli paketler
require(MASS)
attach(Boston)
# Veri seti açiklamasi
?Boston

# Satir sayisi
nrow(Boston)
# Sütun sayisi
ncol(Boston)

# Part b
# Karsilastirma grafigi
pairs(Boston, pch = 21)
# Yüksek korelasyon katsayisina sahip olan özelliklerin grafikleri
indices <- which((abs(cor(Boston))>0.75 & abs(cor(Boston))<1), arr.ind = TRUE)
indices <- unique(indices)
pairs(Boston[,indices],pch = 21)

# Part c
sort(abs(cor(Boston)[,1]), decreasing = TRUE)
pairs(Boston[, c('rad','tax','lstat','nox','indus')])

# Part d
hist(Boston$crim, breaks = 20)
Boston[Boston$crim>40,]

hist(Boston$tax, breaks = 20)
Boston[Boston$tax>700,]

hist(Boston$ptratio, breaks = 50)
Boston[Boston$ptratio>=22,]

# Part e
nrow(Boston[Boston$chas == 1, ])

# Part f
median(Boston$ptratio)

# Part g
Boston[min(Boston$medv),]
summary(Boston)

# part h
nrow(Boston[Boston$rm>7,])
nrow(Boston[Boston$rm>8,])

Boston[Boston$rm>8,]
summary(Boston)

