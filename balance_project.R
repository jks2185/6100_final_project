library(fastDummies)
library(microbenchmark)
library(rpart)
#install.packages("microbenchmark")
library(caret)
library(randomForest)
library(e1071)
library(fastDummies)
source("ROC_func.R")
library(C50)
library(gbm)


balance_data <- read.csv("data/balance-scale.csv")

head(balance_data)
unique(balance_data$class_name)

dim(balance_data)
table(balance_data$class_name)

balance_data$class_name <- ifelse(balance_data$class_name == "B",0,1)

train <- sample(1:nrow(balance_data), 0.7*nrow(balance_data))
validate <- setdiff(1:nrow(balance_data), train)

train_set <- balance_data[train,]
valid_set <- balance_data[validate,]

dt_model <- rpart(class_name ~ ., data = train_set, method = "class")
dt_prediction <- predict(dt_model, newdata = valid_set, type = "class")

dt_prediction

confusionMatrix(dt_prediction, as.factor(valid_set$class_name))$overall['Accuracy']
confusionMatrix(dt_prediction, as.factor(valid_set$class_name))$byClass['Sensitivity']
confusionMatrix(dt_prediction, as.factor(valid_set$class_name))$byClass['Precision']
confusionMatrix(dt_prediction, as.factor(valid_set$class_name))$byClass['F1']

rf_prediction <- randomForest(as.factor(class_name) ~ ., data = train_set)
rf_prediction <- predict(rf_prediction, newdata = valid_set)

rf_prediction

confusionMatrix(rf_prediction, as.factor(valid_set$class_name))$overall['Accuracy']
confusionMatrix(rf_prediction, as.factor(valid_set$class_name))$byClass['Sensitivity']
confusionMatrix(rf_prediction, as.factor(valid_set$class_name))$byClass['Precision']
confusionMatrix(rf_prediction, as.factor(valid_set$class_name))$byClass['F1']

svm_model <- svm(as.factor(class_name) ~ ., data = train_set, kernel = "radial", cost = 1)
svm_prediction <- predict(svm_model, newdata = valid_set)

svm_prediction

confusionMatrix(svm_prediction, as.factor(valid_set$class_name))$overall['Accuracy']
confusionMatrix(svm_prediction, as.factor(valid_set$class_name))$byClass['Sensitivity']
confusionMatrix(svm_prediction, as.factor(valid_set$class_name))$byClass['Precision']
confusionMatrix(svm_prediction, as.factor(valid_set$class_name))$byClass['F1']

c50_model <- C5.0(as.factor(class_name) ~ ., data = train_set)
c50_prediction <- predict(c50_model, newdata = valid_set)

c50_prediction

confusionMatrix(c50_prediction, as.factor(valid_set$class_name))$overall['Accuracy']
confusionMatrix(c50_prediction, as.factor(valid_set$class_name))$byClass['Sensitivity']
confusionMatrix(c50_prediction, as.factor(valid_set$class_name))$byClass['Precision']
confusionMatrix(c50_prediction, as.factor(valid_set$class_name))$byClass['F1']

gbm_model <- gbm(class_name ~ ., data = train_set)
gbm_prediction <- predict(gbm_model, newdata = valid_set, type = "response")

gbm_prediction
gbm_prediction <- ifelse(gbm_prediction > .5,1,0)

confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$class_name))$overall['Accuracy']
confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$class_name))$byClass['Sensitivity']
confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$class_name))$byClass['Precision']
confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$class_name))$byClass['F1']

glm_model <- glm(as.integer(class_name) ~ ., data = train_set)
glm_prediction <- predict(glm_model, newdata = valid_set)

glm_prediction <- ifelse(glm_prediction > .5,1,0)

confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$class_name))$overall['Accuracy']
confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$class_name))$byClass['Sensitivity']
confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$class_name))$byClass['Precision']
confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$class_name))$byClass['F1']


###CROSS VALIDATION###


####CROSS VALIDATION#####

train <- sample(1:nrow(balance_data), 0.7*nrow(balance_data))
validate <- setdiff(1:nrow(balance_data), train)

train_set <- balance_data[train,]
valid_set <- balance_data[validate,]

ctrl <- trainControl(method = "cv", number = 20, verboseIter = TRUE)

dt_timing <- system.time({
  dt_model <- train(class_name ~ ., data = train_set, method = "rpart", trControl = ctrl)
  dt_prediction <- dt_prediction <- predict(dt_model, newdata = valid_set)
})

dt_prediction <- ifelse(dt_prediction > .5,1,0)


rf_timing <- system.time({
  rf_model <- train(class_name ~ ., data = train_set, method = "rf", trControl = ctrl)
  rf_prediction <- rf_prediction <- predict(rf_model, newdata = valid_set)
})

rf_prediction
rf_prediction <- ifelse(rf_prediction > .5,1,0)


svm_timing <- system.time({
  svm_model <- train(class_name ~ ., data = train_set, method = "svmRadial", trControl = ctrl)
  svm_prediction <- svm_prediction <- predict(svm_model, newdata = valid_set)
})
svm_prediction
svm_prediction <- ifelse(svm_prediction > .5,1,0)


c50_timing <- system.time({
  c50_model <- train(as.factor(class_name) ~ ., data = train_set, method = "C5.0", trControl = ctrl)
  c50_prediction <- c50_prediction <- predict(c50_model, newdata = valid_set)
})
c50_prediction


gbm_timing <- system.time({
  gbm_model <- train(as.factor(class_name) ~ ., data = train_set, method = "gbm", trControl = ctrl)
  gbm_prediction <- gbm_prediction <- predict(gbm_model, newdata = valid_set)
})
gbm_prediction


glm_timing <- system.time({
  glm_model <- train(as.factor(class_name) ~ ., data = train_set, method = "glm", trControl = ctrl, family = "binomial")
  glm_prediction <- predict(glm_model, newdata = valid_set)
})
glm_prediction


cat('dt acc: ', confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$class_name))$overall['Accuracy'])
cat('rf acc: ', confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$class_name))$overall['Accuracy'])
cat('svm acc: ', confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$class_name))$overall['Accuracy'])
cat('c50 acc: ', confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$class_name))$overall['Accuracy'])
cat('gbm acc: ', confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$class_name))$overall['Accuracy'])
cat('glm acc: ', confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$class_name))$overall['Accuracy'])



cat('dt sens: ', confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$class_name))$byClass['Sensitivity'])
cat('rf sens: ', confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$class_name))$byClass['Sensitivity'])
cat('svm sens: ', confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$class_name))$byClass['Sensitivity'])
cat('c50 sens: ', confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$class_name))$byClass['Sensitivity'])
cat('gbm sens: ', confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$class_name))$byClass['Sensitivity'])
cat('glm sens: ', confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$class_name))$byClass['Sensitivity'])



cat('dt prec: ', confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$class_name))$byClass['Precision'])
cat('rf prec: ', confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$class_name))$byClass['Precision'])
cat('svm prec: ', confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$class_name))$byClass['Precision'])
cat('c50 prec: ', confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$class_name))$byClass['Precision'])
cat('gbm prec: ', confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$class_name))$byClass['Precision'])
cat('glm prec: ', confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$class_name))$byClass['Precision'])


cat('dt f1: ', confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$class_name))$byClass['F1'])
cat('rf f1: ', confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$class_name))$byClass['F1'])
cat('svm f1: ', confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$class_name))$byClass['F1'])
cat('c50 f1: ', confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$class_name))$byClass['F1'])
cat('gbm f1: ', confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$class_name))$byClass['F1'])
cat('glm f1: ', confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$class_name))$byClass['F1'])

cat('dt time: ', dt_timing['elapsed'])
cat('rf time: ', rf_timing['elapsed'])
cat('svm time: ', svm_timing['elapsed'])
cat('c50 time: ', c50_timing['elapsed'])
cat('gbm time: ', gbm_timing['elapsed'])
cat('glm time: ', glm_timing['elapsed'])
