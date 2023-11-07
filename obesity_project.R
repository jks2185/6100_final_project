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
data$ord_CAEC <- as.integer(factor(data$CAEC, levels = caec_levels, ordered = TRUE))
data$ord_CALC <- as.integer(factor(data$CALC, levels = caec_levels, ordered = TRUE))

data <- dummy_cols(data, select_columns = "MTRANS")

head(data)
drop_cols <- c("CAEC", "CALC", "MTRANS", "NObeyesdad")
data <- data[, !(names(data) %in% drop_cols)]
target_col <-  c("ord_NObeyesdad")
data$ord_NObeyesdad <- as.factor(data$ord_NObeyesdad)
head(data)




#min_max_norm <- function(x){
#  (x - min(x)) / (max(x) - min(x))
#}

#x_cols <- names(pairs_data[, !(names(pairs_data) %in% target_col)])
#x_cols

#data_norm <- as.data.frame(lapply(pairs_data[x_cols], min_max_norm))

library(rpart)
#install.packages("microbenchmark")
library(microbenchmark)
library(caret)
library(randomForest)
library(e1071)
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
rm(dt_accuracy_array, rt_accuracy_array, dt_mean_time_array)
dt_accuracy_array <- c()
dt_mean_time_array <- c()
rt_accuracy_array <- c()


for(i in 1:20){
  
  train <- sample(1:nrow(data), 0.7*nrow(data))
  validate <- setdiff(1:nrow(data), train)
  ####DECISION TREE MODEL####
  timing <- microbenchmark(
  dt_model <- rpart(data$ord_NObeyesdad ~ ., data = data, subset = train, method = "class"),
  prediction <- predict(dt_model, data[validate,], type = "class")
  )
  #print(dt_matrix$overall['Accuracy'])
  dt_mean_time_array <- c(dt_mean_time_array, mean(timing$time)) 
  dt_accuracy_array <- c(dt_accuracy_array, confusionMatrix(prediction, as.factor(data[validate,]$ord_NObeyesdad))$overall['Accuracy'])
  ###DECISION TREE MODEL####
  
  #print(confusionMatrix(rf_prediction, as.factor(data[validate,]$ord_NObeyesdad)))
  ####RANDOM TREE MODEL####
  rf_model <- randomForest(ord_NObeyesdad ~ ., data = data, subset = train, ntree = 100, type = "response")
  rf_prediction <- predict(rf_model, data[validate,], type = "response")
  rt_accuracy_array <- c(rt_accuracy_array, confusionMatrix(rf_prediction, as.factor(data[validate,]$ord_NObeyesdad))$overall['Accuracy'])
  ####RANDOM TREE MODEL####
  
  cat(i,"Done")
}

print(length(dt_accuracy_array))
print(length(rt_accuracy_array))
print(length(dt_mean_time_array))

print(sum(dt_accuracy_array)/length(dt_accuracy_array))
print(sum(rt_accuracy_array)/length(rt_accuracy_array))


train <- sample(1:nrow(data), 0.7*nrow(data))
validate <- setdiff(1:nrow(data), train)

svm_model <- svm(ord_NObeyesdad ~ ., data = data, subset = train, kernel = 'radial', cost = 1)
svm_predict <- predict(svm_model, newdata = data[validate,])

confusionMatrix(svm_predict, as.factor(data[validate,]$ord_NObeyesdad))$byClass

result <- confusionMatrix(svm_predict, as.factor(data[validate,]$ord_NObeyesdad), mode = 'everything')

sum(result$byClass[,3]) / length(result$byClass[,3])
