library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Mayat wanita ditemukan dikardus.', 'Mayat dikardus bernama Rika.',
          'Rika merupakan seorang mualaf.','pelaku pembunuhan merupakan costumer korban.',
          'korban merupakan anak yang tertutup.','korban bekerja digerai kosmetik.',
          'korban sebelumnya akan dilamar oleh sang pacar.',
          'ditemukan di samping Gereja Huria Kristen Batak Protestan (HKBP) Ampera.',
          'pelaku pembunuhan diduga kesal karena barang yang dipesan tak kunjung datang.',
          'korban merupakan seorang keturunan tionghoa muslim.',
          'Mayat wanita didalam kardus ditemukan dalam keadaan telanjang.',
          'pelaku pembunuhan diduga pacar korban.','pacar korban tidak terlihat saat pemakaman',
          'diduga korban merupakan korban penculikan.','sebelumnya korban diduga diperkosa.',
          'korban dibunuh dengan dibacok','pembunuhan diduga karena faktor asmara.',
          'korban pembuhuna dimutilasi','motor yang digunakan merupakan motor pelaku.',
          'mayat korban dimasukkan ke dalam koper.','pelaku pembunuhan merupakan kerabat dekat korban.')
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
data
train
# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c('Pembunuh Rika Karina tertangkap.','Pelaku menghabisi nyawa Rika Karina dengan tusukan pisau.',
           'mayat dimasukkan dalam kardus.','elaku bernama Hendri mengaku kesal dengan korban.',
           'Hendri membenturkan kepala korban, menusuk leher dan menyayat tangan kanan korban dengan menggunakan pisau dapur.',
           'mayat dimasukkan dalam tas.','pelaku merupakan pacar korban.','pembunuhan dilakukan dirumah tersangka.',
           'korban ditemukan dalam keadaan telanjang.','diduga merupakan korban pemerkosaan.')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)