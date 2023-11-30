library(fastDummies)

data <- read.csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')


head(data)
dim(data)

any(is.na(data))

unique(data$MTRANS)

summary(data)

data$Gender <- ifelse(data$Gender == "Male",1,0)
data$family_history_with_overweight <- ifelse(data$family_history_with_overweight == "yes",1,0)
data$SMOKE <- ifelse(data$SMOKE == "yes",1,0)
data$FAVC <- ifelse(data$FAVC == 'yes',1,0)
data$SCC <- ifelse(data$SCC == 'yes',1,0)


table(data$family_history_with_overweight)

head(data)

#data$CAEC <- as.factor(data$CAEC)
#data$CALC <- as.factor(data$CALC)
#data$NObeyesdad <- as.factor(data$NObeyesdad)

unique(data$NObeyesdad)
unique(data$CAEC)
unique(data$CALC)

levels = c("Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III")
caec_levels <- c("no", "Sometimes", "Frequently", "Always")
calc_levels <- c("no", "Sometimes", "Frequently", "Always")

data$ord_NObeyesdad <- as.integer(factor(data$NObeyesdad, levels = levels, ordered = TRUE))
data$ord_CAEC <- as.numeric(factor(data$CAEC, levels = caec_levels, ordered = TRUE))
data$ord_CALC <- as.numeric(factor(data$CALC, levels = caec_levels, ordered = TRUE))


data <- dummy_cols(data, select_columns = "MTRANS")

data$MTRANS_Automobile <- as.numeric(data$MTRANS_Automobile)
data$MTRANS_Bike <- as.numeric(data$MTRANS_Bike)
data$MTRANS_Motorbike <- as.numeric(data$MTRANS_Motorbike)
data$MTRANS_Public_Transportation <- as.numeric(data$MTRANS_Public_Transportation)
data$MTRANS_Walking <- as.numeric(data$MTRANS_Walking)

head(data)
drop_cols <- c("CAEC", "CALC", "MTRANS", "NObeyesdad")
data <- data[, !(names(data) %in% drop_cols)]
target_col <-  c("ord_NObeyesdad")
#data$ord_NObeyesdad <- as.factor(data$ord_NObeyesdad)
head(data)

data$ord_NObeyesdad <- ifelse(data$ord_NObeyesdad > 4,1,0)

data$ord_NObeyesdad

sapply(data, as.integer)
sapply(data, class)

library(rpart)
#install.packages("microbenchmark")
library(microbenchmark)
library(caret)
library(randomForest)
library(e1071)
library(C50)
library(gbm)
library(bsnsing)
# length(prediction)
# length(data$ord_NObeyesdad)


# fit_decision_tree <- function(data, target, train){
#   decision_tree_model <- rpart(target ~ ., data = data, subset = train, method = "class")
#   return(decision_tree_model)
# }
# 
# make_deicision_tree_preidiction <- function(model = NULL, newdata = data[validate,]){
#   prediction <- predict(model, newdata, type = "class")
#   return(prediction)
# }
rm(dt_accuracy_array, rt_accuracy_array, dt_mean_time_array, svm_accuracy_array, c50_accuracy_array)
dt_accuracy_array <- c()
dt_mean_time_array <- c()
rt_accuracy_array <- c()
svm_accuracy_array <- c()
c50_accuracy_array <- c()

train <- sample(1:nrow(data), 0.7*nrow(data))
validate <- setdiff(1:nrow(data), train)

dt_model <- rpart(data$ord_NObeyesdad ~ ., data = data, subset = train, method = "class")
prediction <- predict(dt_model, data[validate,], type = "class")


rf_model <- randomForest(as.factor(ord_NObeyesdad) ~ ., data = data, subset = train, ntree = 100)
rf_prediction <- predict(rf_model, data[validate,], type = "response")


for(i in 1:20){
  train <- sample(1:nrow(data), 0.7*nrow(data))
  validate <- setdiff(1:nrow(data), train)
  ####DECISION TREE MODEL####
  dt_timing <- microbenchmark(
  dt_model <- rpart(data$ord_NObeyesdad ~ ., data = data, subset = train, method = "class"),
  prediction <- predict(dt_model, data[validate,], type = "class")
  )
  #print(dt_matrix$overall['Accuracy'])
  dt_mean_time_array <- c(dt_mean_time_array, mean(dt_timing$time)) 
  dt_accuracy_array <- c(dt_accuracy_array, mean(data$ord_NObeyesdad == prediction))
  ###DECISION TREE MODEL####
  
  #print(confusionMatrix(rf_prediction, as.factor(data[validate,]$ord_NObeyesdad)))
  ####RANDOM TREE MODEL####
  rf_timing <- microbenchmark(
    rf_model <- randomForest(as.factor(ord_NObeyesdad) ~ ., data = data, subset = train, ntree = 100, type = "response"),
    rf_prediction <- predict(rf_model, data[validate,], type = "response")
  )
  rt_accuracy_array <- c(rt_accuracy_array, confusionMatrix(rf_prediction, as.factor(data[validate,]$ord_NObeyesdad))$overall['Accuracy'])
  ####RANDOM TREE MODEL####
  ###SVM MODEL####
  svm_timing <- microbenchmark(
    svm_model <- svm(as.factor(ord_NObeyesdad) ~ ., data = data, subset = train, kernel = 'radial', cost = 1),
    svm_predict <- predict(svm_model, newdata = data[validate,])
  )
  svm_accuracy_array <- c(svm_accuracy_array, confusionMatrix(svm_predict, as.factor(data[validate,]$ord_NObeyesdad))$overall['Accuracy'])
  ###SVM MODEL####
  ###c50 model####
  c50_timing <-microbenchmark(
    c50_model <- C5.0(as.factor(ord_NObeyesdad) ~ ., data = data, subset = train),
    c50_pred <- predict(c50_model, newdata = data[validate,])
  )
  c50_accuracy_array <- c(c50_accuracy_array, mean(data[validate,]$ord_NObeyesdad == c50_pred))
  ###c50 model####
  ###GBM MODEL####
  gbm_timing <- microbenchmark(
    gbm_model <- gbm(ord_NObeyesdad ~ ., data = data[train,]),
    gbm_predict <- predict(gbm_model, newdata = data[validate,], type = "response")
  )
  gbm_predict <- ifelse(gbm_predict > .5,1,0)
  gbm_accuracy <- mean(gbm_predict == data[validate,]$ord_NObeyesdad)
  ###GBM MODEL###
  ###BSNSING MODEL###
  bsnsing_timing <- microbenchmark(
    bsnsing_model <- bsnsing(as.integer(ord_NObeyesdad) ~ ., data = data, subset = train),
    bsnsing_predict <- predict(bsnsing_model, newdata = data[validate,])
  )
  
  cat(i,"Done")
}


train <- sample(1:nrow(data), 0.7*nrow(data))
validate <- setdiff(1:nrow(data), train)


gbm_model <- gbm(ord_NObeyesdad ~ ., data = data[train,])
gbm_predict <- predict(gbm_model, newdata = data[validate,], type = "response")

bsnsing_model <- bsnsing(as.integer(ord_NObeyesdad) ~ ., data = data, subset = train)
bsnsing_predict <- predict(bsnsing_model, newdata = data[validate,])

bsnsing_predict <- ifelse(bsnsing_predict > .5,1,0)
bsnsing_accuracy <- mean(bsnsing_predict == data[validate,]$ord_NObeyesdad)
bsnsing_accuracy

gbm_accuracy
