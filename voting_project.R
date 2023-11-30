voting_data <- read.csv('data/house-votes-84.csv')

class(voting_data)

head(voting_data)

sapply(voting_data, class)
voting_data[, sapply(voting_data, is.character)] <- lapply(voting_data[, sapply(voting_data, is.character)], as.factor)
voting_data$class_name <- ifelse(voting_data$class_name == 'democrat',1,0)

train <- sample(1:nrow(voting_data), 0.7*nrow(voting_data))
validate <- setdiff(1:nrow(voting_data), train)

dt_model <- rpart(class_name ~ ., data = voting_data, subset = train, method = "class")
prediction <- predict(dt_model, voting_data[validate,], type = "class")

prediction

mean(prediction == voting_data[validate,]$class_name)

rf_model <- randomForest(as.factor(class_name) ~ ., data = voting_data, subset = train, ntree = 100)
rf_prediction <- predict(rf_model, voting_data[validate,], type = "response")

mean(rf_prediction == voting_data[validate,]$class_name)

svm_model <- svm(as.factor(class_name) ~ ., data = voting_data, subset = train, kernel = 'radial', cost = 1)
svm_predict <- predict(svm_model, newdata = voting_data[validate,])

svm_predict
mean(svm_predict == voting_data[validate,]$class_name)

c50_model <- C5.0(as.factor(class_name) ~ ., data = voting_data, subset = train)
c50_pred <- predict(c50_model, newdata = voting_data[validate,])

c50_pred
mean(c50_pred == voting_data[validate,]$class_name)

gbm_model <- gbm(class_name ~ ., data = voting_data[train,])
gbm_predict <- predict(gbm_model, newdata = voting_data[validate,], type = "response")

gbm_predict
gbm_predict <- ifelse(gbm_predict > .5,1,0)
mean(gbm_predict == voting_data[validate,]$class_name)

bsnsing_model <- bsnsing(as.factor(class_name) ~ ., data = voting_data, subset = train)
bsnsing_predict <- predict(bsnsing_model, newdata = voting_data[validate,])

bsnsing_predict
bsnsing_predict <- ifelse(bsnsing_predict > .5,1,0)
mean(bsnsing_predict == voting_data[validate,]$class_name)
