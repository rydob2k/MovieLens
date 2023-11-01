
# LOAD packages:
library(tidyverse)
if(!require(recosystem)) {install.packages("recosystem"); library(recosystem)}

if(!require(doMC)) {install.packages("doMC"); library(doMC)} # for multi-core processing
registerDoMC(cores = 10)

load("~/Projects/MovieLens/rda/edx.RData")
load("./rda/test_edx.RData")
load("./rda/train_edx.RData")

# Convert the train and test data sets into input format for recosystem
set.seed(50)
train_rec <- with(train_edx, data_memory(user_index = userId,
                                         item_index = movieId,
                                         rating = rating))
test_rec <- with(test_edx, data_memory(user_index = userId,
                                         item_index = movieId,
                                         rating = rating))

# Assign the RecoSys object model:
rec <- recosystem::Reco()

# Train the model with default tuning parameters
mat_fit <- rec$train(train_rec)

# Predict the ratings for the test set with the matrix factorization model
pred_matrix <- rec$predict(test_rec, out_memory())

# Calculate the RMSE
matrix_fact_rmse <- RMSE(test_edx$rating, pred_matrix)
