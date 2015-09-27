# Machine Learning Course Project

# Load the libraries

library(caret)
library(dplyr)
library(doMC)

registerDoMC(cores = 3)


# Set the seed for reproducibility

set.seed(45876)

# Set the training and test data URLs
# Note: Changed to http for now; look at https and read.csv

trainLink <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

testLink <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Get the data 

trainData <- read.csv(url(trainLink))

testData <- read.csv(url(testLink))

# Remove all columns with NA

colNAS <- colSums(is.na(trainData))

nonNACols <- colNAS[colNAS == 0]

colNames <- names(nonNACols)

trainDataExtract <- trainData[,colNames]

# Remove the X value which is the row number of the data set

trainDataExtract$X <- NULL
testData$X <- NULL

# Test the extract set for near zero variance

nsvExtract <- nearZeroVar(trainDataExtract)

# Remove the non zero variance columns

trainDataFinal <- trainDataExtract[, -nsvExtract]

# Training Set Histogram

plot01 <- ggplot(trainDataFinal) + 
            geom_histogram(aes(x = classe, fill= classe)) +
            ggtitle("Training Data Set - Exercise Classes") +
            scale_x_discrete(labels = c("Sitting Down", "Standing Up", "Standing", "Walking", "Sitting")) +
            xlab("Exercise Class") +
            ylab("Count") +
            scale_fill_discrete(name="Exercise Class", labels=c("Sitting Down", "Standing Up", "Standing", "Walking", "Sitting"))

print(plot01)

# Training control

trainCTRL <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# Test the decision tree model

if(file.exists("decision_tree_train.RData")) {
      
      load("decision_tree_train.RData")
      
} else {
      
      modelDTFit <- train(classe ~ ., data = trainDataFinal, method = "rpart", trControl = trainCTRL)
      save(modelDTFit, file = "decision_tree_train.RData")
}

# Test the random forest model

if(file.exists("random_forest_train.RData")) {
      
      load("random_forest_train.RData")
      
} else {
      
      modelRFFit <- train(classe ~ ., data = trainDataFinal, method = "rf", trControl = trainCTRL)
      save(modelRFFit, file = "random_forest_train.RData")
}


# Test the decision tree model

dtResult <- predict(modelDTFit, trainDataFinal, type = "raw")

# Create the confusion matrix

dtMatrix <- confusionMatrix(dtResult, trainDataFinal$classe )

# Test the random forest model

rfResult <- predict(modelRFFit, trainDataFinal, type = "raw")

rfMatrix <- confusionMatrix(rfResult, trainDataFinal$classe )

# Perform the actual test

testResult <- predict(modelRFFit, testData)

# Function for the 20 test files

pml_write_files = function(x) {
      
      n = length(x)
      
      for(i in 1:n) {
            
            filename = paste0("problem_id_",i,".txt")
            write.table(x[i],file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
            
      }
      
}

results <- as.character(testResult)

pml_write_files(results)
