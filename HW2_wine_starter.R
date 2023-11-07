# Execute the lines below to prepare data
wine <- read.csv("winequality-red.csv")
wine$quality <- as.factor(wine$quality)
set.seed(2023)
train <- sample(1:nrow(wine), 0.7*nrow(wine))
valid <- setdiff(1:nrow(wine), train)
input_vars <- colnames(wine)[1:11]

# Scratch area: Put your R code for exploratory analysis below. If you don't know what to do on your own, follow the tutorial PDF on Canvas and try some commands there. The scratch code does not need to be well organized or annotated.  


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

