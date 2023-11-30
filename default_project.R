cc_data <- read.csv('data/default of credit card clients.csv')

head(cc_data)

any(is.na(cc_data))

unique(cc_data$LIMIT_BAL)

sum(cc_data$LIMIT_BAL) / length(cc_data$LIMIT_BAL)




dim(cc_data)

library(microbenchmark)
library(rpart)
#install.packages("microbenchmark")
library(caret)
library(randomForest)
library(e1071)

train <- sample(1:nrow(cc_data), 0.7*nrow(cc_data))
validate <- setdiff(1:nrow(cc_data), train)



dt_model <- rpart(cc_data$default.payment.next.month ~ ., data = cc_data, subset = train, method = "class")
dt_prediction <- predict(dt_model, cc_data[validate,], type = "class")

mean(dt_prediction == cc_data[validate,]$default.payment.next.month)

rf_model <- randomForest(as.factor(default.payment.next.month) ~ ., data = cc_data, subset = train, ntree = 100, type = "response")
rf_prediction <- predict(rf_model, cc_data[validate,], type = "response")

rf_prediction
mean(rf_prediction == cc_data[validate,]$default.payment.next.month)

svm_model <- svm(as.factor(default.payment.next.month) ~ ., data = cc_data, subset = train, kernel = 'radial', cost = 1)
svm_predict <- predict(svm_model, newdata = cc_data[validate,])

svm_predict
mean(svm_predict == cc_data[validate,]$default.payment.next.month)

c50_model <- C5.0(as.factor(default.payment.next.month) ~ ., data = cc_data, subset = train)
c50_pred <- predict(c50_model, newdata = cc_data[validate,])

c50_pred
mean(c50_pred == cc_data[validate,]$default.payment.next.month)

gbm_model <- gbm(default.payment.next.month ~ ., data = cc_data[train,])
gbm_predict <- predict(gbm_model, newdata = cc_data[validate,], type = "response")

gbm_predict
gbm_predict <- ifelse(gbm_predict > .5,1,0)
mean(gbm_predict == cc_data[validate,]$default.payment.next.month)

bsnsing_model <- bsnsing(as.integer(default.payment.next.month) ~ ., data = cc_data, subset = train)
bsnsing_predict <- predict(bsnsing_model, newdata = data[validate,])

bsnsing_predict


for(i in 1:20){
  dt_accuracy_array <- c()
  dt_mean_time_array <- c()
  rt_accuracy_array <- c()
  
  train <- sample(1:nrow(cc_data), 0.7*nrow(cc_data))
  validate <- setdiff(1:nrow(cc_data), train)
  ####DECISION TREE MODEL####
  timing <- microbenchmark(
    dt_model <- rpart(cc_data$default.payment.next.month ~ ., data = cc_data, subset = train, method = "class"),
    prediction <- predict(dt_model, cc_data[validate,], type = "class")
  )
  #print(dt_matrix$overall['Accuracy'])
  dt_mean_time_array <- c(dt_mean_time_array, mean(timing$time)) 
  dt_accuracy_array <- c(dt_accuracy_array, confusionMatrix(prediction, as.factor(cc_data[validate,]$default.payment.next.month))$overall['Accuracy'])
  ###DECISION TREE MODEL####
  
  
  ####RANDOM TREE MODEL####
  rf_model <- randomForest(as.factor(cc_data$default.payment.next.month) ~ ., data = cc_data, subset = train, ntree = 100)
  rf_prediction <- predict(rf_model, cc_data[validate,], type = "response")
  rt_accuracy_array <- c(rt_accuracy_array, confusionMatrix(rf_prediction, as.factor(cc_data[validate,]$default.payment.next.month))$overall['Accuracy'])
  ####RANDOM TREE MODEL####
  
  cat(i,"Done")
}


