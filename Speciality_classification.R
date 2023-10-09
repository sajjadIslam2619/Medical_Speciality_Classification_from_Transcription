# Author: K M Sajjadul Islam

#install.packages("NLP")
#install.packages("Matrix")
#install.packages("ROSE")
#install.packages("MLSMOTE")
#install.packages("smotefamily")
#install.packages("lsa")
#install.packages("SnowballC")
#install.packages("textreuse")
#install.packages("text2vec")
#install.packages("caret")

##clear your working directory
rm(list=ls())
# Load the Matrix package
library(Matrix)
library(tm)
library(dplyr)
library(e1071)
library(graphics)
library(ROSE)
library(smotefamily)
library(lsa)
library(textreuse)
library(text2vec)
library(caret)

# Data can be found in: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
data<-read.csv(file = 'mtsamples.csv', header=TRUE, stringsAsFactors=FALSE)

df <- data[c("transcription", "medical_specialty")]
#length(df$medical_specialty)

# Remove row with empty value.
df <- na.omit(df)
#length(df$medical_specialty)

# Count each 'medical_specialty' category. 40 category found. In few categories observation is very low, such as "Hospice - Palliative Care = 6"
data.frame(table(df$medical_specialty))

# Remove category/class with observations less than n (100/200/500) 
df_filtered <- df %>% 
  group_by(medical_specialty) %>% 
  filter(n() >= 200)

# After filtering class less than than n (100, 200, 500) observations, got y (11 ,8, 2) category/class  
data.frame(table(df_filtered$medical_specialty))

# Remove leading and trailing spaces from column 'medical_specialty'
df_filtered$medical_specialty <- trimws(df_filtered$medical_specialty)


######## Up-sampling selected specialty by duplicating each observation twice. ######
######## Select 8 class and up-sample 6 lowest observed classes. ####################
######## Accuracy increased from around 10% (from 30% to 40%) #######################

df_filtered_Gastroenterology <- subset(df_filtered,  medical_specialty == "Gastroenterology")
df_filtered <- rbind(df_filtered, df_filtered_Gastroenterology)

df_filtered_General_Medicine <- subset(df_filtered,  medical_specialty == "General Medicine")
df_filtered <- rbind(df_filtered, df_filtered_General_Medicine)

df_filtered_Neurology <- subset(df_filtered,  medical_specialty == "Neurology")
df_filtered <- rbind(df_filtered, df_filtered_Neurology)

df_filtered_Radiology <- subset(df_filtered,  medical_specialty == "Radiology")
df_filtered <- rbind(df_filtered, df_filtered_Gastroenterology)

df_filtered_Cardiovascular <- subset(df_filtered,  medical_specialty == "Cardiovascular / Pulmonary")
df_filtered <- rbind(df_filtered, df_filtered_Cardiovascular)

df_filtered_Orthopedic <- subset(df_filtered,  medical_specialty == "Orthopedic")
df_filtered <- rbind(df_filtered, df_filtered_Orthopedic)

data.frame(table(df_filtered$medical_specialty))


############ Data pre-processing ############

# Remove only digit words from 'transcription' column
df_filtered$transcription <- gsub("[[:digit:]]+", "", df_filtered$transcription)
#Load in a corpus
transcription_corpus <- VCorpus(VectorSource(df_filtered$transcription))
#inspect(transcription_corpus[[1]])
#Convert to lower case
transcription_corpus <- tm_map(transcription_corpus, content_transformer(tolower))
#Remove punctuation
transcription_corpus <- tm_map(transcription_corpus, removePunctuation)
#Remove stop words
transcription_corpus <- tm_map(transcription_corpus, removeWords, stopwords())
#inspect(transcription_corpus[[1]])

df_filtered$clean_transcription <- sapply(transcription_corpus, as.character)


########## Down-sampling 'Surgery' by using cosine similarity measure. ####################
######### 'Surgery' is the largest class. Remove observations which are 70% similar #######
######### Down-sampling has no significant effect on accuracy #############################

# df_filtered_Surgery <- subset(df_filtered,  medical_specialty == "Surgery")

# Measure cosine similarity takes lots of time. For testing purpose taking subset of 'surgery' class.
# df_filtered_Surgery <- sample(df_filtered_Surgery, n = 100)

# df_filtered <- df_filtered[-which(df_filtered$medical_specialty == "Surgery"),]
# data.frame(table(df_filtered$medical_specialty))
# 
# surgery_dup_row_list <- list()
# df_filtered_Surgery <- df_filtered_Surgery[nchar(df_filtered_Surgery$clean_transcription) >= 10, ]
# 
# for (i in 1:nrow(df_filtered_Surgery)) {
#   for (j in (i+1):nrow(df_filtered_Surgery)){
#     text1 <- df_filtered_Surgery[i,3]
#     text2 <- df_filtered_Surgery[j,3]
#     if (length(text1) == 0 | length(text2) == 0) {
#       next
#     }
# 
#     # create a document-term matrix using the 'text2vec' package
#     tokens <- word_tokenizer(c(text1, text2))
#     it <- itoken(tokens, progressbar = FALSE)
#     vocab <- create_vocabulary(it)
#     vectorizer <- vocab_vectorizer(vocab)
#     dtm <- create_dtm(it, vectorizer)
# 
#     # convert the document-term matrix to a sparse matrix
#     dtm_sparse <- as(dtm, "CsparseMatrix")
# 
#     # calculate the cosine similarity between the two text strings
#     cos_sim <- cosine(dtm_sparse[1, ], dtm_sparse[2, ])
#     cos_sim <- as.numeric(cos_sim)
# 
#     if (!is.na(cos_sim) && is.finite(cos_sim) && cos_sim > 0.7) {
#       surgery_dup_row_list <- append(surgery_dup_row_list, i)
#     }
#   }
# }
# for (i in surgery_dup_row_list) {
#   df_filtered_Surgery <- df_filtered_Surgery[-i, ]
# }
# df_filtered <- rbind(df_filtered, df_filtered_Surgery)
# data.frame(table(df_filtered$medical_specialty))

# Change class ('medical_specialty') to numerical value.
df_filtered$medical_specialty <- as.integer(factor(df_filtered$medical_specialty, levels = unique(df_filtered$medical_specialty)))
data.frame(table(df_filtered$medical_specialty))

transcription_corpus <- VCorpus(VectorSource(df_filtered$clean_transcription))

########## Predict only with Naive Bayes ###############

#Generate TF-IDF matrix
transcription_dtm <- DocumentTermMatrix(transcription_corpus)
#findFreqTerms(transcription_dtm,5)

# Remove terms not in 90% of the documents. 80% or 100% decreases the accuracy
dense_transcription_dtm <- removeSparseTerms(transcription_dtm, .90)

# Inspect to TF-IDF
#inspect(dense_transcription_dtm)

# Convert the document term matrix to a matrix
transcription_matrix <- as.matrix(dense_transcription_dtm)

# Split the data into training and testing sets
train.index <- sample(1:nrow(transcription_matrix), nrow(transcription_matrix) * 0.8)
train <- transcription_matrix[train.index, ]
test <- transcription_matrix[-train.index, ]

labels <- factor(df_filtered$medical_specialty[train.index])

# Train a Naive Bayes classifier
nb <- naiveBayes(train, labels)

# Predict the labels for the testing data
predictions <- predict(nb, test)
actual <- df_filtered$medical_specialty[-train.index]

# Calculate the accuracy of the classifier
accuracy <- sum(predictions == df_filtered$medical_specialty[-train.index]) / length(predictions)
print(paste0("Accuracy: ", round(accuracy * 100, 2), "%"))

# Create a confusion matrix
confusionMatrix(factor(actual), factor(predictions))

################# Implement PCA and then predict with Naive Bayes ###################
############ PCA decreases accuracy around 5% #######################################
transcription_dtm <- DocumentTermMatrix(transcription_corpus)
dtm <- removeSparseTerms(transcription_dtm, sparse = 0.9) 

dtm_matrix <- as.matrix(dtm)
dtm_matrix <- scale(dtm_matrix)
# Apply PCA to the document-term matrix
pca <- prcomp(dtm_matrix, scale = TRUE)
#biplot(prcomp(dtm_matrix, scale = TRUE))

# Get the PCA data
transformed.matrix <- pca$x

# Split the data into training and testing sets
train.index <- sample(1:nrow(transformed.matrix), nrow(transformed.matrix) * 0.8)
train <- transformed.matrix[train.index, ]
test <- transformed.matrix[-train.index, ]

# Create a factor vector for the labels
labels <- factor(df_filtered$medical_specialty[train.index])

# Train a Naive Bayes classifier
nb <- naiveBayes(train, labels)

# Predict the labels for the testing data
predictions <- predict(nb, test)

# Calculate the accuracy of the classifier
accuracy <- sum(predictions == df_filtered$medical_specialty[-train.index]) / length(predictions)
print(paste0("Accuracy: ", round(accuracy * 100, 2), "%"))
