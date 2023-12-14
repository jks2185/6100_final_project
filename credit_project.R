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



credit_data <- read.csv("data/crx.csv")

head(credit_data)


credit_data <- dummy_cols(credit_data, select_columns = c("a6", "a1", "a4", "a5", "a7", "a9", "a10", "a12", "a13"))


names(credit_data) <- gsub(x = names(credit_data), pattern = "\\-", replacement = "_")
names(credit_data) <- gsub(x = names(credit_data), pattern = "\\?", replacement = "U")
names(credit_data) <- gsub(x = names(credit_data), pattern = "\\(", replacement = "_")
names(credit_data) <- gsub(x = names(credit_data), pattern = "\\)", replacement = "_")
names(credit_data) <- gsub(x = names(credit_data), pattern = "\\&", replacement = "_")

drop_cols <- c("a6", "a1", "a4", "a5", "a7", "a9", "a10", "a12", "a13", "a4_U", "a5_U")
credit_data <- credit_data[, !(names(credit_data) %in% drop_cols)]


credit_data$a14 <- as.numeric(credit_data$a14)
credit_data$a2 <- as.numeric(credit_data$a2)
credit_data$a16 <- ifelse(credit_data$a16 == "+",1,0)
any(is.na(credit_data))
which(is.na(credit_data))

credit_data <- na.omit(credit_data)
dim(credit_data)
sapply(credit_data, class)

dt_accuracy_array <- c()
dt_mean_time_array <- c()
dt_sensitivity_array <- c()
dt_precision_array <- c()
dt_F1_array <- c()

rf_accuracy_array <- c()
rf_mean_time_array <- c()
rf_sensitivity_array <- c()
rf_precision_array <- c()
rf_F1_array <- c()

svm_accuracy_array <- c()
svm_mean_time_array <- c()
svm_sensitivity_array <- c()
svm_precision_array <- c()
svm_F1_array <- c()

c50_accuracy_array <- c()
c50_mean_time_array <- c()
c50_sensitivity_array <- c()
c50_precision_array <- c()
c50_F1_array <- c()

gbm_accuracy_array <- c()
gbm_mean_time_array <- c()
gbm_sensitivity_array <- c()
gbm_precision_array <- c()
gbm_F1_array <- c()

glm_accuracy_array <- c()
glm_mean_time_array <- c()
glm_sensitivity_array <- c()
glm_precision_array <- c()
glm_F1_array <- c()


for(i in 1:20){
  train <- sample(1:nrow(credit_data), 0.7*nrow(credit_data))
  validate <- setdiff(1:nrow(credit_data), train)
  
  train_set <- credit_data[train,]
  valid_set <- credit_data[validate,]
  
  dt_timing <- system.time({
    dt_model <- rpart(a16 ~ ., data = train_set, method = "class")
    dt_prediction <- predict(dt_model, newdata = valid_set, type = "class")
  })
  
  
  dt_mean_time_array <- c(dt_mean_time_array, dt_timing["elapsed"]) 
  dt_accuracy_array <- c(dt_accuracy_array, confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
  dt_sensitivity_array <- c(dt_sensitivity_array, confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
  dt_precision_array <- c(dt_precision_array, confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$a16))$byClass['Precision'])
  dt_F1_array <- c(dt_F1_array, confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$a16))$byClass['F1'])
  
  rf_timing <- system.time({
    rf_model <- randomForest(as.factor(a16) ~ ., data = train_set)
    rf_prediction <- predict(rf_model, newdata = valid_set)
  })
  
  rf_mean_time_array <- c(rf_mean_time_array, rf_timing["elapsed"]) 
  rf_accuracy_array <- c(rf_accuracy_array, confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
  rf_sensitivity_array <- c(rf_sensitivity_array, confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
  rf_precision_array <- c(rf_precision_array, confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$a16))$byClass['Precision'])
  rf_F1_array <- c(rf_F1_array, confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$a16))$byClass['F1'])
  
  svm_timing <- system.time({
    svm_model <- svm(as.factor(a16) ~ ., data = train_set, kernel = "radial", cost = 1)
    svm_prediction <- predict(svm_model, newdata = valid_set)
  })
  
  svm_mean_time_array <- c(svm_mean_time_array, svm_timing["elapsed"]) 
  svm_accuracy_array <- c(svm_accuracy_array, confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
  svm_sensitivity_array <- c(svm_sensitivity_array, confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
  svm_precision_array <- c(svm_precision_array, confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$a16))$byClass['Precision'])
  svm_F1_array <- c(svm_F1_array, confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$a16))$byClass['F1'])
  
  c50_timing <- system.time({
    c50_model <- C5.0(as.factor(a16) ~ ., data = train_set)
    c50_prediction <- predict(c50_model, newdata = valid_set)
  })
  
  c50_mean_time_array <- c(c50_mean_time_array, c50_timing["elapsed"]) 
  c50_accuracy_array <- c(c50_accuracy_array, confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
  c50_sensitivity_array <- c(c50_sensitivity_array, confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
  c50_precision_array <- c(c50_precision_array, confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$a16))$byClass['Precision'])
  c50_F1_array <- c(c50_F1_array, confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$a16))$byClass['F1'])
  
  gbm_timing <- system.time({
    gbm_model <- gbm(as.integer(a16) ~ ., data = train_set)
    gbm_prediction <- predict(gbm_model, newdata = valid_set, type = "response")
  })
  
  gbm_prediction <- ifelse(gbm_prediction > .5,1,0)
  
  gbm_mean_time_array <- c(gbm_mean_time_array, gbm_timing["elapsed"]) 
  gbm_accuracy_array <- c(gbm_accuracy_array, confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
  gbm_sensitivity_array <- c(gbm_sensitivity_array, confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
  gbm_precision_array <- c(gbm_precision_array, confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$a16))$byClass['Precision'])
  gbm_F1_array <- c(gbm_F1_array, confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$a16))$byClass['F1'])
  
  glm_timing <- system.time({
    glm_model <- glm(as.integer(a16) ~ ., data = train_set)
    glm_prediction <- predict(glm_model, newdata = valid_set)
  })
  
  glm_prediction <- ifelse(glm_prediction > .5,1,0)
  
  glm_mean_time_array <- c(glm_mean_time_array, glm_timing["elapsed"]) 
  glm_accuracy_array <- c(glm_accuracy_array, confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
  glm_sensitivity_array <- c(glm_sensitivity_array, confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
  glm_precision_array <- c(glm_precision_array, confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$a16))$byClass['Precision'])
  glm_F1_array <- c(glm_F1_array, confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$a16))$byClass['F1'])
}



dt_time_average <- (sum(dt_mean_time_array) / length(dt_mean_time_array))
dt_accuracy_average <- (sum(dt_accuracy_array) / length(dt_accuracy_array))
dt_sensitivity_average <- (sum(dt_sensitivity_array) / length(dt_sensitivity_array))
dt_precision_average <- (sum(dt_precision_array) / length(dt_precision_array)) 
dt_f1_average <- (sum(dt_F1_array) / length(dt_F1_array))

cat("dt time: ",dt_time_average)
cat("Dt acc: ", dt_accuracy_average)
cat("dt sens: ",dt_sensitivity_average)
cat("dt prec: ", dt_precision_average)
cat("dt f1: ", dt_f1_average)

rf_time_average <- (sum(rf_mean_time_array) / length(rf_mean_time_array))
rf_accuracy_average <- (sum(rf_accuracy_array) / length(rf_accuracy_array))
rf_sensitivity_average <- (sum(rf_sensitivity_array) / length(rf_sensitivity_array))
rf_precision_average <- (sum(rf_precision_array) / length(rf_precision_array)) 
rf_f1_average <- (sum(rf_F1_array) / length(rf_F1_array))

cat("rf time: ",rf_time_average)
cat("rf acc: ", rf_accuracy_average)
cat("rf sens: ",rf_sensitivity_average)
cat("rf prec: ", rf_precision_average)
cat("rf f1: ", rf_f1_average)

svm_time_average <- (sum(svm_mean_time_array) / length(svm_mean_time_array))
svm_accuracy_average <- (sum(svm_accuracy_array) / length(svm_accuracy_array))
svm_sensitivity_average <- (sum(svm_sensitivity_array) / length(svm_sensitivity_array))
svm_precision_average <- (sum(svm_precision_array) / length(svm_precision_array)) 
svm_f1_average <- (sum(svm_F1_array) / length(svm_F1_array))

cat("svm time: ",svm_time_average)
cat("svm acc: ", svm_accuracy_average)
cat("svm sens: ",svm_sensitivity_average)
cat("svm prec: ", svm_precision_average)
cat("svm f1: ", svm_f1_average)

c50_time_average <- (sum(c50_mean_time_array) / length(c50_mean_time_array))
c50_accuracy_average <- (sum(c50_accuracy_array) / length(c50_accuracy_array))
c50_sensitivity_average <- (sum(c50_sensitivity_array) / length(c50_sensitivity_array))
c50_precision_average <- (sum(c50_precision_array) / length(c50_precision_array)) 
c50_f1_average <- (sum(c50_F1_array) / length(c50_F1_array))

cat('c50 time:', c50_time_average)
cat('c50 accuracy:', c50_accuracy_average)
cat('c50 sensitivity:', c50_sensitivity_average)
cat('c50 precision:', c50_precision_average)
cat('c50 f1:', c50_f1_average)

gbm_time_average <- (sum(gbm_mean_time_array) / length(gbm_mean_time_array))
gbm_accuracy_average <- (sum(gbm_accuracy_array) / length(gbm_accuracy_array))
gbm_sensitivity_average <- (sum(gbm_sensitivity_array) / length(gbm_sensitivity_array))
gbm_precision_average <- (sum(gbm_precision_array) / length(gbm_precision_array)) 
gbm_f1_average <- (sum(gbm_F1_array) / length(gbm_F1_array))

cat('gbm time:', gbm_time_average)
cat('gbm accuracy:', gbm_accuracy_average)
cat('gbm sensitivity:', gbm_sensitivity_average)
cat('gbm precision:', gbm_precision_average)
cat('gbm f1:', gbm_f1_average)

glm_time_average <- (sum(glm_mean_time_array) / length(glm_mean_time_array))
glm_accuracy_average <- (sum(glm_accuracy_array) / length(glm_accuracy_array))
glm_sensitivity_average <- (sum(glm_sensitivity_array) / length(glm_sensitivity_array))
glm_precision_average <- (sum(glm_precision_array) / length(glm_precision_array)) 
glm_f1_average <- (sum(glm_F1_array) / length(glm_F1_array))

cat('glm time:', glm_time_average)
cat('glm accuracy:', glm_accuracy_average)
cat('glm sensitivity:', glm_sensitivity_average)
cat('glm precision:', glm_precision_average)
cat('glm f1:', glm_f1_average)


####CROSS VALIDATION#####

train <- sample(1:nrow(credit_data), 0.7*nrow(credit_data))
validate <- setdiff(1:nrow(credit_data), train)

train_set <- credit_data[train,]
valid_set <- credit_data[validate,]

ctrl <- trainControl(method = "cv", number = 20, verboseIter = TRUE)

dt_timing <- system.time({
  dt_model <- train(a16 ~ ., data = train_set, method = "rpart", trControl = ctrl)
  dt_prediction <- dt_prediction <- predict(dt_model, newdata = valid_set)
})

dt_prediction <- ifelse(dt_prediction > .5,1,0)


rf_timing <- system.time({
  rf_model <- train(a16 ~ ., data = train_set, method = "rf", trControl = ctrl)
  rf_prediction <- rf_prediction <- predict(rf_model, newdata = valid_set)
})

rf_prediction
rf_prediction <- ifelse(rf_prediction > .5,1,0)


svm_timing <- system.time({
  svm_model <- train(a16 ~ ., data = train_set, method = "svmRadial", trControl = ctrl)
  svm_prediction <- svm_prediction <- predict(svm_model, newdata = valid_set)
})
svm_prediction
svm_prediction <- ifelse(svm_prediction > .5,1,0)


c50_timing <- system.time({
  c50_model <- train(as.factor(a16) ~ ., data = train_set, method = "C5.0", trControl = ctrl)
  c50_prediction <- c50_prediction <- predict(c50_model, newdata = valid_set)
})
c50_prediction


gbm_timing <- system.time({
  gbm_model <- train(as.factor(a16) ~ ., data = train_set, method = "gbm", trControl = ctrl)
  gbm_prediction <- gbm_prediction <- predict(gbm_model, newdata = valid_set)
})
gbm_prediction


glm_timing <- system.time({
  glm_model <- train(as.factor(a16) ~ ., data = train_set, method = "glm", trControl = ctrl, family = "binomial")
  glm_prediction <- predict(glm_model, newdata = valid_set)
})
glm_prediction


cat('dt acc: ', confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
cat('rf acc: ', confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
cat('svm acc: ', confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
cat('c50 acc: ', confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
cat('gbm acc: ', confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$a16))$overall['Accuracy'])
cat('glm acc: ', confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$a16))$overall['Accuracy'])



cat('dt sens: ', confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
cat('rf sens: ', confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
cat('svm sens: ', confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
cat('c50 sens: ', confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
cat('gbm sens: ', confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])
cat('glm sens: ', confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$a16))$byClass['Sensitivity'])



cat('dt prec: ', confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$a16))$byClass['Precision'])
cat('rf prec: ', confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$a16))$byClass['Precision'])
cat('svm prec: ', confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$a16))$byClass['Precision'])
cat('c50 prec: ', confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$a16))$byClass['Precision'])
cat('gbm prec: ', confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$a16))$byClass['Precision'])
cat('glm prec: ', confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$a16))$byClass['Precision'])


cat('dt f1: ', confusionMatrix(as.factor(dt_prediction), as.factor(valid_set$a16))$byClass['F1'])
cat('rf f1: ', confusionMatrix(as.factor(rf_prediction), as.factor(valid_set$a16))$byClass['F1'])
cat('svm f1: ', confusionMatrix(as.factor(svm_prediction), as.factor(valid_set$a16))$byClass['F1'])
cat('c50 f1: ', confusionMatrix(as.factor(c50_prediction), as.factor(valid_set$a16))$byClass['F1'])
cat('gbm f1: ', confusionMatrix(as.factor(gbm_prediction), as.factor(valid_set$a16))$byClass['F1'])
cat('glm f1: ', confusionMatrix(as.factor(glm_prediction), as.factor(valid_set$a16))$byClass['F1'])

cat('dt time: ', dt_timing['elapsed'])
cat('rf time: ', rf_timing['elapsed'])
cat('svm time: ', svm_timing['elapsed'])
cat('c50 time: ', c50_timing['elapsed'])
cat('gbm time: ', gbm_timing['elapsed'])
cat('glm time: ', glm_timing['elapsed'])