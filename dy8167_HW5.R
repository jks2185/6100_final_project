load("winequality-red-1.RData")
# MODIFY the line below: use the last two digits of you access ID, e.g., gn0061 gives 61
set.seed(61)  # <-- MODIFY this number
# DO NOT MODIFY the next four lines
wine$quality <- ifelse(wine$quality >= 6, 1, 0)  # 1 means good quality, 0 means bad quality
trainset <- sample(1:nrow(wine), 1000)  # DO NOT CHANGE: you must sample 1000 data points for training
validset <- setdiff(1:nrow(wine), trainset)  # The remaining is used for validation
source("ROC_func.R")  # source in the ROC_func.R (presumably located in your current directory)

wine$quality <- as.integer(wine$quality)
train <- wine[trainset,]
valid <- wine[validset,]

sapply(wine, class)

#RPART
rp <- rpart::rpart(quality ~., data = train)
rpart.plot::rpart.plot(rp)
pred_rp <- predict(rp, newdata = wine[validset,])

pred_rp

mean(wine$quality == factor_rp)

#BSNGING
bs <- bsnsing::bsnsing(quality ~., data = train)
pred_bs <- predict(bs, newdata = wine[validset,])

pred_bs

#brif
bf <- brif::brif(quality ~., data = train)
pred_bf_matrix <- predict(bf, newdata = wine[validset,])
pred_bf <- pred_bf_matrix$`1`
pred_bf <- pred_bf_matrix[,"1"]

pred_bf

#C50
c50_model <- C50::C5.0(as.factor(quality) ~ ., data = train)
c50_pred <- predict(c50_model, newdata = wine[validset,])

c50_pred


#party:ctree
ctree_model <- party::ctree(as.factor(quality) ~ ., data = train)
ctree_predict <- predict(ctree_model, newdata = wine[validset,])

#tree::tree
tree_model <- tree::tree(quality ~ ., data = train)
tree_predict <- predict(tree_model, newdata = wine[validset,])

#randomForest::randomForest
rf_model <- randomForest::randomForest(quality ~ ., data = train)
rf_predict <- predict(rf_model, newdata = wine[validset,])

rf_predict


#xgboost::xgboost
dtrain <- xgboost::xgb.DMatrix(data = as.matrix(train[, -train$quality]), label = train$quality)
dtest <- xgboost::xgb.DMatrix(data = as.matrix(valid[, -valid$quality]), label = valid$quality)
params <- list(
  objective = "binary:logistic",
  max_depth = 3,
  eta = 0.1,
  eval_metric = "logloss"
)
xgboost_model <- xgboost::xgboost(data = dtrain, params = params, nrounds = 100)
xgboost_predict <- predict(xgboost_model, newdata = dtest)

#gbm::gbm
gbm_model <- gbm::gbm(quality ~ ., data = train)
gbm_predict <- predict(gbm_model, newdata = wine[validset,], type = "response")

gbm_predict

df <- data.frame(true.label = wine[validset, 'quality'],
                 pred_rp = pred_rp,
                 pred_bs = pred_bs,
                 pred_bf = pred_bf,
                 pred_c50 = c50_pred,
                 ctree = ctree_predict,
                 tree = tree_predict,
                 rf = rf_predict,
                 xgb = xgboost_predict,
                 gbm = gbm_predict)

pdf("ROC Curve.pdf", width = 16, height = 14)
auc_rp = ROC_func(df, 1, 2, color = 'black')
auc_bs = ROC_func(df, 1, 3, color = 'red', add_on = T)
auc_bf = ROC_func(df, 1, 4, color = 'blue', add_on = T)
auc_c50 = ROC_func(df, 1, 5, color = 'green', add_on = T)
auc_Ctree = ROC_func(df, 1, 6, color = 'orange', add_on = T)
auc_tree = ROC_func(df, 1, 7, color = 'grey', add_on = T)
auc_rf = ROC_func(df, 1, 8, color = 'pink', add_on = T)
auc_xgb = ROC_func(df, 1, 9, color = 'purple', add_on = T)
auc_gbm = ROC_func(df, 1, 10, color = 'violet', add_on = T)

legend("bottomright", legend = c('rpart',
                                 'bsnsing',
                                 'brif',
                                 'c50',
                                 'ctree',
                                 'tree',
                                 'randomForest',
                                 'xgboost',
                                 'gbm'),
       col = c('black','red','blue', 'green', 'orange', 'grey', 'pink', 'purple', 'violet'),
       lty = 1)
dev.off()

