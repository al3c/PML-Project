## practical machine learning - course project
## data from this website:
## http://groupware.les.inf.puc-rio.br/har
#setwd("/Volumes/Livingston/Coursera - Data Science/8 - Practical Machine Learning")
library(caret)
library(knitr)
library(rpart)
library(data.table)

## loading data in
raw <- as.data.table(read.csv("./project data/pml-training.csv", header=TRUE, 
                              sep=","))
raw <- raw[,X := NULL] # counter row
# raw[,classe] <- as.factor(raw[,classe])
# for (i in 6:159) raw[,i] <- as.numeric(raw[,i])

final.testing <- read.csv("./project data/pml-testing.csv", header=TRUE, sep=",",
                          colClasses = "numeric")

## data slicing
set.seed(4444)
inTrain <- createDataPartition(y=raw$classe, p=0.6, list=F)
training <- raw[c(inTrain)]
testing <- raw[-c(inTrain)]
dim(training); dim(testing)

## cleaning
## find columns with NAs
na.cols <- training[,sapply(.SD, function(x) any(is.na(x)))]
# names.na.cols <- names(which(na.cols == T))
# col.remove <- which(training[,substr(colnames(training),1,30) %in% 
#                                  names(which(na.cols == T))])

training <- training[,c(names(which(na.cols == T))) := NULL]
training <- training[,c(1:6) := NULL]

drop.cols <- function (x) {
    
    
}

## exploratory analysis
M <- abs(cor(training[,-86, with=F]))) ## training would have to be data.frame
diag(M) <- 0
which(M > 0.8, arr.ind=T)

## route 1: use NZV to prune predictors, then do PCA
nzv <- nearZeroVar(training, saveMetrics=T)
nzv
var.remove <- rownames(nzv[nzv$nzv==TRUE,])
training <- training[,c(var.remove) := NULL] # leaves only 53 variables (52 pred.)
preProc <- preProcess(training[,-53, with=F], method="pca") # further reduces pred.
trainPC <- predict(preProc, training[,-53, with=F])

## route 2: directly preprocess with PCA
# prComp <- prcomp(training[,-86, with=F])
preProc <- preProcess(training[,-86, with=F], method="pca")

## build model
model.rpart <- train(training$classe ~ ., method = "rpart", data = trainPC)
model.rf <- train(training$classe ~ ., method = "rf", data = trainPC) ## had to 
## kill after taking too long
model.rf <- randomForest(training$classe ~ ., data = trainPC) ## THIS ONE!
model.treebag <- train(training$classe ~ ., method = "treebag", data = trainPC)
model.rda <- train(training$classe ~ ., method = "rda", data = trainPC)

## clean and preprocess testing data
testing <- testing[,c(names(which(na.cols == T))) := NULL]
testing <- testing[,c(1:6) := NULL]
testing <- testing[,c(var.remove) := NULL] 
testPC <- predict(preProc, testing[,-53, with=F])

## apply to test data
# rpart method
confusionMatrix(testing$classe, predict(model.rpart, testPC))

## random forest method
confusionMatrix(testing$classe, predict(model.rf, testPC))
## 97% accurate

## bagged CART
confusionMatrix(testing$classe, predict(model.treebag, testPC))

## RDA
confusionMatrix(testing$classe, predict(model.rda, testPC))

## variable importance
plot(importance(model.rf))
# plot(varImp(model.rf))

## submitting predictions
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("./answers/problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(predictions)

write.pml.predictions <- function(x) {
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}