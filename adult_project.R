library("stringr")

adult_data <- read.csv('data/adult.csv', check.names = TRUE)

head(adult_data)

unique(adult_data$occupation)
unique(adult_data$marital_status)
unique(adult_data$work_class)
unique(adult_data$relationship)
unique(adult_data$race)
unique(adult_data$country)
unique(adult_data$sex)
unique(adult_data$income)

adult_data$sex <- ifelse(adult_data$sex == 'Male',1,0)
adult_data$income <- ifelse(adult_data$income == ' >50K',1,0)

adult_data <- dummy_cols(adult_data, select_columns = "occupation")
adult_data <- dummy_cols(adult_data, select_columns = "marital_status")
adult_data <- dummy_cols(adult_data, select_columns = "work_class")
adult_data <- dummy_cols(adult_data, select_columns = "relationship")
adult_data <- dummy_cols(adult_data, select_columns = "race")
adult_data <- dummy_cols(adult_data, select_columns = "country")

names(adult_data) <- sub(" ", "", names(adult_data))

drop_cols <- c("occupation","marital_status","work_class","relationship","race","education","country")
adult_data <- adult_data[, !(names(adult_data) %in% drop_cols)]



head(adult_data)

names(adult_data) <- gsub(x = names(adult_data), pattern = "\\-", replacement = "_")
names(adult_data) <- gsub(x = names(adult_data), pattern = "\\?", replacement = "U")
names(adult_data) <- gsub(x = names(adult_data), pattern = "\\(", replacement = "_")
names(adult_data) <- gsub(x = names(adult_data), pattern = "\\)", replacement = "_")
names(adult_data) <- gsub(x = names(adult_data), pattern = "\\&", replacement = "_")



train <- sample(1:nrow(adult_data), 0.7*nrow(adult_data))
validate <- setdiff(1:nrow(adult_data), train)

dt_model <- rpart(income ~ ., data = adult_data, subset = train, method = "class")
dt_prediction <- predict(dt_model, adult_data[validate,], type = "class")

dt_prediction
mean(dt_prediction == adult_data[validate,]$income)


rf_model <- randomForest(income ~ ., data = adult_data, subset = train, ntree = 100)
rf_prediction <- predict(rf_model, adult_data[validate,], type = "response")

rf_prediction
mean(rf_prediction == adult_data[validate,]$income)


svm_model <- svm(income ~ ., data = adult_data, subset = train, kernel = 'radial', cost = 1)
svm_predict <- predict(svm_model, newdata = adult_data[validate,])

svm_predict
svm_predict <- ifelse(svm_predict > .5,1,0)
mean(svm_predict == adult_data[validate,]$income)

c50_model <- C5.0(as.factor(income) ~ ., data = adult_data, subset = train)
c50_pred <- predict(c50_model, newdata = adult_data[validate,])

c50_pred
mean(c50_pred == adult_data[validate,]$income)

gbm_model <- gbm(income ~ ., data = adult_data[train,])
gbm_predict <- predict(gbm_model, newdata = adult_data[validate,], type = "response")

gbm_predict
gbm_predict <- ifelse(gbm_predict > .5,1,0)
mean(gbm_predict == adult_data[validate,]$income)

bsnsing_model <- bsnsing(as.integer(income) ~ ., data = adult_data, subset = train)
bsnsing_predict <- predict(bsnsing_model, newdata = adult_data[validate,])

bsnsing_predict
bsnsing_predict <- ifelse(bsnsing_predict > .5,1,0)
mean(bsnsing_predict == adult_data[validate,]$income)

