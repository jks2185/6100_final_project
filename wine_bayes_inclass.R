wine <- read.csv("winequality-red.csv")
wine$good <- ifelse(wine$quality > 5, 1, 0)
summary(wine$good)

par(mfrow = c(1,2))
hist(wine[which(wine$good == 1), 'volatile.acidity'])
hist(wine[which(wine$good == 0), 'volatile.acidity'])
par(mfrow = c(1,1))

mu_good <- mean(wine[which(wine$good == 1), 'volatile.acidity'])
mu_bad <- mean(wine[which(wine$good == 0), 'volatile.acidity'])
sd_good <- sd(wine[which(wine$good == 1), 'volatile.acidity'])
sd_bad <- sd(wine[which(wine$good == 0), 'volatile.acidity'])
prior_good <- mean(wine$good)
prior_bad <- 1 - prior_good

# new wine has volatile acidity 0.8
prior_good * dnorm(0.8, mu_good, sd_good)
prior_bad * dnorm(0.8, mu_bad, sd_bad)
# so the prediction is "bad"

BayesModel <- list(mu_good = mu_good,
                   sd_good = sd_good,
                   mu_bad = mu_bad,
                   sd_bad = sd_bad,
                   prior_good = prior_good,
                   prior_bad = prior_bad
                   )

predictBayes <- function(myModel, newdata){
  n <- nrow(newdata)
  pred <- integer(n)
  for(i in 1:n){
    post_good <- myModel$prior_good * dnorm(newdata[i,'volatile.acidity'], myModel$mu_good, myModel$sd_good)
    post_bad <- myModel$prior_bad * dnorm(newdata[i,'volatile.acidity'], myModel$mu_bad, myModel$sd_bad)
    pred[i] <- ifelse(post_good > post_bad, 1, 0)
  }
  return(pred)
}

predictBayes(myModel = BayesModel, newdata = data.frame(volatile.acidity = c(0.8, 1.5, 0.2)))


# rebuild the model on train set and test it on valid set
train <- sample(1:nrow(wine), 0.5*nrow(wine))
valid <- setdiff(1:nrow(wine), train)

mu_good <- mean(wine[intersect(which(wine$good == 1), train), 'volatile.acidity'])
mu_bad <- mean(wine[intersect(which(wine$good == 0), train), 'volatile.acidity'])
sd_good <- sd(wine[intersect(which(wine$good == 1), train), 'volatile.acidity'])
sd_bad <- sd(wine[intersect(which(wine$good == 0), train), 'volatile.acidity'])
prior_good <- mean(wine[train, 'good'])
prior_bad <- 1 - prior_good

# model built on train set
BayesModel <- list(mu_good = mu_good,
                   sd_good = sd_good,
                   mu_bad = mu_bad,
                   sd_bad = sd_bad,
                   prior_good = prior_good,
                   prior_bad = prior_bad
)

pred <- predictBayes(myModel = BayesModel, newdata = wine[valid, ])
accuracy <- sum(pred == wine[valid,'good'])/length(valid)
# or
mean(pred == wine[valid,'good'])
