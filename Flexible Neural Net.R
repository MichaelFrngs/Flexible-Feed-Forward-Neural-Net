setwd("C:/Users/mfrangos2016/Desktop/Neural Network")
#> install.packages("DMwR") #Allows the "unscale" function
#> install.packages("neuralnet")
library(MASS)
library(neuralnet)


########INPUTS#########

#Sets Target Variable (variable we're predicting) 
#Set this to the name of the column you're predicting.
TargetVariable = "z" #You can enter two variables using c("x1",x2"). 

NeuralNetNodeStructure =  c(10,5) #first hidden layer is 4 nodes, second hidden layer is 2 nodes
WhereToSplitDataSet = 7     #Which row should we split the data into two sets (training/test set)

InitialDataSet = read.csv("Data.csv")

#There's a "K" variable on line 105 that will set the number of folds in the cross validation.

########END OF INPUTS#########



#Setting a seed makes sure you get the same numbers every time.
set.seed(123)

#Gives you information on the dataset we're using
help("Boston")
str(Boston)
dim(InitialDataSet) #Shows dimensions of the InitialDataSet

#Lets take a look at the target variable, or what we're trying to predict
hist(InitialDataSet[,TargetVariable]) #We see that the data is left skewed

#The apply function applies a function to either columns(1)/rows(2)/both(c(1,2)). In this case, we're getting the range of every column
apply(InitialDataSet,2,range) #Range returns the minimum & the maximum. We see that the scale of each variable is not the same. 

#We want to normalize the data, in other words, replace the values with something from 0 to 1. [Called normalized data in interval [0,1]]
#Normalizing data prevents any one column from dominating the others.

#Get the max of every column
maxValue = apply(InitialDataSet,2,max)
#Get the min of every column
minValue = apply(InitialDataSet,2,min)

#Normalizes the data. Now everything is between 0 and 1.
InitialDataSet.normalized = as.data.frame(scale(InitialDataSet,center = minValue,scale = maxValue - minValue))

#Gives us the list of column names
allVars = colnames(InitialDataSet.normalized)
#Creates a random sample row-index of our data, sized at 400 observations.
data.sample.rowindex = sample(1:nrow(InitialDataSet),WhereToSplitDataSet)
saved.X.Values = InitialDataSet[-data.sample.rowindex,!allVars%in%TargetVariable]
#Create training set & test set using a sample of the data
trainDF= InitialDataSet.normalized[data.sample.rowindex,]
testDF = InitialDataSet.normalized[-data.sample.rowindex,] #Somehow, this gives us the unused dataset. Not sure how.




#we'll merge this with our predictions later. Reads as "all vars that are not the target varaible.
testDF.x.values = testDF[,!allVars%in%TargetVariable] 

#Read as: The predictor vars are ALL Variables that are NOT the target variable (in this case "medv")
predictorVars = allVars[!allVars%in%TargetVariable] #<----Notice you leave out the comma in the index when using arrays. Gives error...

#Let's place a plus sign between all of those names we got to make using the neural net easier.
predictorVars = paste(predictorVars, collapse = "+") #Collapse is different than sep parameter...

#Creates the formula in the form of: TargetVar ~ Var1 + Var2 + Var3..... ETC. It says TargetVar is a function of all this other stuff.
FormYooLah = as.formula(paste(paste(TargetVariable,  collapse="+"),"~",predictorVars,collapse = "+")) 

#Neural net is the function we use for fitting
neuralModel = neuralnet(formula = FormYooLah,  
                        hidden = NeuralNetNodeStructure, #We're using a 13 - 4 - 2 - 1 structure. 13 vars, and 4 to 2 nodes, to 1 output node.
                        linear.output = TRUE,
                        data = trainDF,
                        stepmax=1e+07,   #If your neural network is not convering, you must make this higher. Increase calculation time exponentially.
                        threshold=0.0001) #Lower is more accurate. If your neural network is not convering, you must make this higher. Does not increase calculation time. 

#You can take a look at the neural network to see if it worked properly. 
plot(neuralModel)

#compute(InitialDataSet, predictorVariables) #Notice the results are still scaled and bounded from 0 to 1
predictions = neuralnet::compute(neuralModel,testDF[,!allVars%in%TargetVariable]) #neuralnet:: prevents a function-name conflict between packages

str(predictions)

#UNSCALES (Reverses Data Normalization)
#Gets the final node (the prediction), which is the column named "net.result".
predictions = predictions$net.result*(max(InitialDataSet[,TargetVariable])-min(InitialDataSet[,TargetVariable])) + min(InitialDataSet[,TargetVariable])

#UNSCALES
#We compare predictions to these actual values to see how good our model is.
actualValues = (testDF[,TargetVariable]) * (max(InitialDataSet[,TargetVariable]) - min(InitialDataSet[TargetVariable])) + min(InitialDataSet[,TargetVariable])



ResultsCombined = data.frame(predictions,actualValues,saved.X.Values)

#Test3 = *(max(InitialDataSet[,TargetVariable]) - min(InitialDataSet[TargetVariable])) + min(InitialDataSet[,TargetVariable])

#Compare predictions to actual results. MSE = sum(x.prediction - x.actual)^2 / observations
MeanSquaredError = sum((predictions - actualValues)^2)/nrow(testDF)

#Print MSE. We see it's almost 0!!
MeanSquaredError

#Plot how well our predictions are to the actual results.
plot(x=testDF[,TargetVariable],y=predictions,
     col = "blue",
     main = "Real vs Predicted",
     pch = 1,      #Point graphics. What will the point on the graph look like.
     cex = 0.9,    #Text size
     type = "p",
     xlab ="Actual",
     ylab ="Predicted")



###########CROSS VALIDATION SECTION
set.seed(450)
CrossValidationMeanSquaredError.Array <- NULL #Initialize this variable since its being used in the loop.
k <- 10  #Number of folds in the cross validation.

library(plyr) 

#Creates Progress bar... cuz neural nets could take some time.
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){                        
  index <- sample(1:nrow(InitialDataSet),round((1-(1/k))*nrow(InitialDataSet)))
  train.cv <- InitialDataSet.normalized[index,]
  test.cv <- InitialDataSet.normalized[-index,]
  
  nn <- neuralnet(FormYooLah,
                  data=train.cv,
                  hidden=NeuralNetNodeStructure,
                  linear.output=T, #Set to false if neural network is not converging
                  stepmax=1e+07,   #If your neural network is not convering, you must make this higher. Increase calculation time exponentially.
                  threshold=0.5) #If your neural network is not convering, you must make this higher. Does not increase calculation time.
  
  #Makes predictions on the testing set using ONLY the [predictor variables]
  pr.nn <- compute(nn,test.cv[,!allVars%in%TargetVariable])
  pr.nn <- pr.nn$net.result*(max(InitialDataSet[,TargetVariable])-min(InitialDataSet[,TargetVariable]))+min(InitialDataSet[,TargetVariable])
  
  test.cv.r <- (test.cv[,TargetVariable])*(max(InitialDataSet[,TargetVariable])-min(InitialDataSet[,TargetVariable]))+min(InitialDataSet[,TargetVariable])
  
  CrossValidationMeanSquaredError.Array[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}

#Print - Gives the MSE of each fold(aka iteration)
CrossValidationMeanSquaredError.Array
#Print the average MSE
mean(CrossValidationMeanSquaredError.Array)

boxplot(CrossValidationMeanSquaredError.Array,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)




####################### OUTCOME CALCULATOR ########################################

#Obviously, the # of parameters entered must match the number of input variables.
#FEED PARAMETERS INTO THIS FUNCTION using c(param.1,param.2, etc). For example: Neural.Predict(c(1,2)) will predict the outcome at point (1,2).
Neural.Predict = function(parameters){
  i=1 #Initialize Variable
  normalized.input=matrix(0,ncol=length(parameters), nrow = 1) #Initialize Matrix with correct length (same number as parameters)
  
  while (i<=length(parameters)) {
    normalized.input[,i] = (parameters[i] - minValue[i]) / (maxValue[i] - minValue[i]) #Normalizes the input parameters, crushing them between 0 & 1.
    names(normalized.input) = names(testDF[,!allVars%in%TargetVariable]) #Sets names of the parameters to equal the variables
  
    i=i+1
  }
  normalized.output = compute(neuralModel,normalized.input) #output, but it's still nonmeaningful because it's normalized between 1 & 0.
  
  #reverses normalization for a meaningful value.
  descaled.output = normalized.output$net.result*(max(InitialDataSet[,TargetVariable])-min(InitialDataSet[,TargetVariable])) + min(InitialDataSet[,TargetVariable])

  paste("The Neural network predicts",descaled.output)
}

