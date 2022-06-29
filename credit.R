library(tidyverse)
library(GGally)
library(psych)
library(car)
library(PRROC)
library(randomForest)
library(caret)
library(mlbench)
library(xgboost)
library(Matrix)
library(MLmetrics)
library(smotefamily)
library(ROSE)

credit_train <- read_csv('C:/Users/685685/Downloads/Data Analysis Using R/Project/credit_train.csv')

credit<- credit_train %>%
  mutate(Class = recode_factor(Class, zero = 'no', one = 'yes'))

table(credit$Class)

# SMOTE credit_train
sample_credit <- SMOTE(X = credit[, -1],  
                       target = credit$Class, 
                       dup_size = 50)
sample_credit_data <- sample_credit$data 
sample_credit_data$class <- factor(sample_credit_data$class) 
table(sample_credit_data$class)
sample_credit_under <- ovun.sample(class ~ .,
                                   data = sample_credit_data, 
                                   method = "under",
                                   N = 213606) 
sample_credit_under_data <- sample_credit_under$data 
table(sample_credit_under_data$class)
credit.smote <- sample_credit_under_data %>% select(class,everything())
colnames(credit.smote)[1] <- 'Class'
table(credit.smote$Class)
credit.smote$Class <- ifelse(credit.smote$Class == 'no', 0, 1)
table(credit.smote$Class)
credit.smote$Class <- factor(credit.smote$Class)

inTrain <- createDataPartition(
  y = credit.smote$Class,
  p = .67,
  list = FALSE
)
train <- credit[inTrain, ]
test <- credit[-inTrain, ]



# training model
# choose best parameters to maximize AUPRC
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  classProbs = TRUE,
  returnResamp = "all",
  summaryFunction = prSummary
)
tunegrid <- expand.grid(
  nrounds = c(50, 100, 200),
  max_depth = c(3, 5),
  eta = c(0.05, 0.1, 0.15, 0.3),
  gamma = c(0,0.2,0.5),
  colsample_bytree = c(0.6,1),
  min_child_weight = c(1,5),
  subsample = c(0.6,1)
)
set.seed(1234)
xgb <- train(
  Class ~ ., 
  data = train, 
  method = 'xgbTree', 
  metric = 'AUC', 
  tuneGrid = tunegrid, 
  trControl = ctrl,
  verbose = FALSE,
  verbosity = 0
)
print(xgb)
plot(xgb)

# choose best thresholds to minimize cost
traind <- data.matrix(train[,-1])
traind <- Matrix(traind, sparse = T)
train_y <- as.numeric(train$Class) - 1
dtrain <- list(data = traind, label = train_y)
dtrain <- xgb.DMatrix(dtrain$data, label = train_y)

testd <- data.matrix(test[,-1])
testd <- Matrix(testd, sparse = T)
test_y <- as.numeric(test$Class) - 1
dtest <- list(data = testd, label = test_y)
dtest <- xgb.DMatrix(dtest$data, label = test_y)

bst <- xgboost(
  dtrain, 
  nrounds = 200,
  eta = 0.3,
  gamma = 0.2,
  max_depth = 5,
  min_child_weight = 5,
  max_delta_step = 0,
  subsample = 1,
  colsample_bytree = 1,
  colsample_bylevel = 1,
  lambda = 1,
  lambda_bias = 0,
  alpha = 1,
  scale_pos_weight = 1,
  objective = 'binary:logistic',
  base_score = 0.5,
  eval_metric = 'aucpr'
)

preds <- predict(bst, dtest)

# Precision-Recall curve of dtest
xgb_scores <- data.frame(event_prob = preds, labels = test_y)
xgb_auprc <- pr.curve(scores.class0 = xgb_scores[xgb_scores$labels == 1, ]$event_prob,
                      # scores for the POSITIVE class
                      scores.class1 = xgb_scores[xgb_scores$labels == 0, ]$event_prob,
                      # scores for the NEGATIVE class
                      curve=T)
plot(xgb_auprc)

cost <- rep(0,990)
test_y <- as.factor(test_y)
threshold.l <- seq(summary(preds)[[1]] + 0.0001, summary(preds)[[6]] - 0.0001, length = 990)
conf.c <- 1000000

for(i in threshold.l){
  preds.th <- rep(0, length(preds))
  preds.th[preds > i] <- 1
  preds.th <- as.factor(preds.th)
  cf.t <- confusionMatrix(preds.th, test_y)$table
  cost.th <- cf.t[1,2] * 145.61 + cf.t[2,1] * 2.6
  cost[which(threshold.l == i)] <- cost.th
  if (conf.c > cost.th) {
    conf.c <- cost.th
    conf.m <- confusionMatrix(preds.th, test_y)
  }
}
(min_cost <- min(cost))
(threshold <- threshold.l[which.min(cost)])
conf.m



# final model
train.bst <- data.matrix(credit.smote[,-1])
train.bst <- Matrix(train.bst, sparse = T)
train_y.bst <- as.numeric(credit.smote$Class) - 1
dtrain.bst <- list(data = train.bst, label = train_y.bst)
dtrain.bst <- xgb.DMatrix(dtrain.bst$data, label = train_y.bst)

xgb.bst <- xgboost(
  dtrain.bst, 
  nrounds = 200,
  eta = 0.3,
  gamma = 0.2,
  max_depth = 5,
  min_child_weight = 5,
  max_delta_step = 0,
  subsample = 1,
  colsample_bytree = 1,
  colsample_bylevel = 1,
  lambda = 1,
  lambda_bias = 0,
  alpha = 1,
  scale_pos_weight = 1,
  objective = 'binary:logistic',
  base_score = 0.5,
  eval_metric = 'aucpr'
)

pred.bst <- data.matrix(credit.smote[,-1])
pred.bst <- Matrix(pred.bst, sparse = T)
dpred.bst <- list(data = pred.bst)
dpred.bst <- xgb.DMatrix(pred.bst)
dpred.bst <- predict(xgb.bst, dpred.bst)

# Precision-Recall curve of whole credit_train
xgb_scores.bst <- data.frame(event_prob = dpred.bst, labels = credit.smote$Class)
xgb_auprc.bst <- pr.curve(scores.class0 = xgb_scores.bst[xgb_scores.bst$labels == 1, ]$event_prob,
                      # scores for the POSITIVE class
                      scores.class1 = xgb_scores.bst[xgb_scores.bst$labels == 0, ]$event_prob,
                      # scores for the NEGATIVE class
                      curve=T)
plot(xgb_auprc.bst)

# prediction
credit_test <- read_csv('C:/Users/685685/Downloads/Data Analysis Using R/Project/credit_test.csv')
pred <- data.matrix(credit_test)
pred <- Matrix(pred, sparse = T)
dpred <- list(data = pred)
dpred <- xgb.DMatrix(pred)
xgb.pred <- predict(xgb.bst, dpred, type = 'response')
Class <- rep('zero', nrow(credit_test))
Class[xgb.pred > threshold] = 'one'
write.csv(Class, 'C:/Users/685685/Downloads/Data Analysis Using R/Project/Class.csv', row.names = FALSE)
