# Execute the lines below to prepare data

wine <- read.csv("winequality-red.csv")
wine$quality <- as.factor(wine$quality)
set.seed(2023)
train <- sample(1:nrow(wine), 0.7*nrow(wine))
valid <- setdiff(1:nrow(wine), train)
input_vars <- colnames(wine)[1:11]

typeof(train)
head(train)

# Scratch area: Put your R code for exploratory analysis below. If you don't know what to do on your own, follow the tutorial PDF on Canvas and try some commands there. The scratch code does not need to be well organized or annotated.  
unique(wine$quality)
class(wine$quality)
table(wine$quality)

dim(wine)
colnames(wine)


hist(wine$density)
barplot(table(wine$quality), xlab = 'quality', ylab = 'obs.')
with(wine, boxplot(fixed.acidity ~ quality))

wine$good <- ifelse(wine$quality > 6, 1, 0)

(formula.string <- paste0(colnames(wine), " ~ quality"))

pdf("wine.numeric_vs_quality.pdf")
for(i in 1:(length(formula.string)-1)){
  with(wine, boxplot(as.formula(formula.string[i])))
}
dev.off()

test <- 'volatile.acidity'

mu_3 <- mean(wine[intersect(which(wine$quality == 3), train), test])
mu_4 <- mean(wine[intersect(which(wine$quality == 4), train), test])
mu_5 <- mean(wine[intersect(which(wine$quality == 5), train), test])
mu_6 <- mean(wine[intersect(which(wine$quality == 6), train), test])
mu_7 <- mean(wine[intersect(which(wine$quality == 7), train), test])
mu_8 <- mean(wine[intersect(which(wine$quality == 8), train), test])

# Write a function to predict wine quality using any subset of the input variables
# The function should return the predicted quality value for each row in newdata
# Do not alter the function declaration, i.e., function name and the argument list
myPredictFunc <- function(model = NULL, newdata = wine[valid, input_vars]){
  ## write your function body here
  ## model can be any object, but it should not encode data from the valid set
  ## Do not reference any information beyond what is packed in the model object
  

  
  return(rep(0,nrow(newdata)))  # placeholder, replace with something meaningful 
}

# Pack your model in an R object called myModel
myModel <- list()  # This is a placeholder, change it

# Test the prediction function on the valid set
pred <- myPredictFunc(myModel)

# Report the accuracy 
(accuracy <- sum(pred == wine[valid,'quality'])/length(valid))






