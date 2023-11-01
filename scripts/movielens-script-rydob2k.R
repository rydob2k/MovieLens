### MOVIELENS PROJECT SCRIPT - RYDOB2K ###

## PURPOSE: The purpose of this project was to develop a model for a movie recommendation system
  # The project was an adaptation based on the Netflix Prize competition hosted by Netflix, in which the goal was to develop the best algorithm for predicting user ratings for films.
  # Algorithms were evaluated based on their root mean squared error (RMSE) 


## Load packages

if(!require(tidyverse)) {install.packages("tidyverse", repos = "http://cran.us.r-project.org"); library(tidyverse)}
if(!require(caret)) {install.packages("caret", repos = "http://cran.us.r-project.org"); library(caret)}
if(!require(qdapTools)) {install.packages("qdapTools", repos = "http://cran.us.r-project.org"); library(qdapTools)}
if(!require(recosystem)) {install.packages("recosystem", repos = "http://cran.us.r-project.org"); library(recosystem)}

options(timeout = 120)

# Optional parallel core processing package:
if(!require(doMC)) {install.packages("doMC"); library(doMC)} 
registerDoMC(cores = 10)

## IMPORT DATA

# Note: data import and partitioning below adapted from HarvardX PH125.9x Project Overview: MovieLens
  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download MovieLens 10M file and unzip to component files: ratings_file and movies_file
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)



## TIDY AND WRANGLE DATA

# Organize ratings_file as data frame, name and classify variables/columns
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Repeat for movies_file, add column names and classes
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")



## PARTITION DATA SETS: training, testing, and final_holdout_test

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Remove non-relevant objects prior to analysis
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Save data objects within RStudio project folder
save(edx, file = "rda/edx.RData")
save(final_holdout_test, file = "rda/final_holdout_test.RData")

# Partition edx into training and test sets
set.seed(10)
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_edx <- edx[-edx_test_index,]
temp2 <- edx[edx_test_index,]

# Make sure userId and movieId in test set are also in train set to avoid NAs
test_edx <- temp2 %>% 
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp2, test_edx)
train_edx <- rbind(train_edx, removed)

# Clean up unused object from environment
rm(edx_test_index, temp2, removed)


## ROOT MEAN SQUARED ERROR
# Create 'RMSE' function for comparing model success
RMSE <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}



## DATA EXPLORATION

# Overview of Training Data - MovieLens 10M
tibble(edx) # table with variables
str(edx)
edx %>% summarize(Users = n_distinct(userId),
                  Movies = n_distinct(movieId))
# 69878 users
# 10677 movies

# Ratings overview by USER
user_ratings <- edx %>% group_by(userId) %>% 
  summarize(n_ratings = n(), median_rating = median(rating), avg_rating = round(mean(rating), digits = 1))
user_ratings

# Skewed distribution of n_ratings data: half of users rated less than 62 movies but mean is 129 movies
user_ratings %>% summarise(average = mean(n_ratings), median = median(n_ratings), min = min(n_ratings), max = max(n_ratings))
user_ratings %>% filter(n_ratings <= (0.10)*max(n_ratings)) %>% 
  ggplot(aes(n_ratings)) + geom_histogram(binwidth = 10) +
  ggtitle("Distribution of Number of Ratings per User")

# Proportion users rating less movies than mean = 129, proportion = 0.72867
sum(user_ratings$n_ratings <= 129) / length(user_ratings$n_ratings)

# Large variation in avg_rating suggests potential effect on model
user_ratings %>% filter(n_ratings >= 30) %>% ggplot(aes(avg_rating)) + geom_histogram(bins = 20) +
  ggtitle("Average Movie Rating per User")


# Ratings overview by MOVIE
movie_ratings <- edx %>% group_by(movieId) %>% 
  summarize(title = title[1], i_ratings = n(), avg_i_rating = round(mean(rating), digits = 1))

# Highest rated movies are obscure titles with less than 5 reviews, so number of ratings affects avg rating
slice_max(movie_ratings, order_by = avg_i_rating, n = 10)

# Only 6% movies were rated greater than 4 stars
sum(movie_ratings$avg_i_rating >= 4) / length(movie_ratings$avg_i_rating)
sum(movie_ratings$avg_i_rating <= 2) / length(movie_ratings$avg_i_rating)
histogram(movie_ratings$avg_i_rating)



## CONSTRUCT BASELINE MODEL: rating = avg_rating + user_effect + movie_effect + genre_effect
# Y = mu + bu + bi + bg

# For baseline model:
mu <- mean(train_edx$rating)

# and RMSE = 1.0598106
base_0_rmse <- RMSE(test_edx$rating, mu)


# Model USER_EFFECT: Initial pass on full training set, where user effect is simply difference between user's average rating and mu
# Compute user average rating and find difference from mu (bu_1 = Y - mu)
bu_1 <- train_edx %>% group_by(userId) %>%
  summarize(bu_1 = mean(rating - mu))

# Predict rating from mu and user effect bu_1 (Y = mu + bu_1)
pred_bu_1 <- test_edx %>%
  left_join(bu_1, by = "userId") %>% 
  mutate(pred = mu + bu_1) %>%
  pull(pred)

# And calculate RMSE
base_1_RMSE <- RMSE(test_edx$rating, pred_bu_1)


# Model MOVIE_EFFECT: Initial pass on full training set, where movie effect is difference between movie's average rating and mu + user_effect
# Compute movie average rating and find difference from mu + bu_1 (bi_1 = Y - mu - bu_1)
bi_1 <- train_edx %>% 
  group_by(movieId) %>%
  left_join(bu_1, by = "userId") %>% 
  summarize(bi_1 = mean(rating - mu - bu_1))

# Predict rating from mu and user effect bu_1 (Y = mu + bu_1)
pred_bu_bi_1 <- test_edx %>%
  left_join(bu_1, by = "userId") %>%
  left_join(bi_1, by = "movieId") %>%
  mutate(pred = mu + bu_1 + bi_1) %>% 
  pull(pred)

# And calculate RMSE
base_2_RMSE <- RMSE(test_edx$rating, pred_bu_bi_1)


# Model GENRE_EFFECT: Initial pass, full training set, where genre effect is genre column as factor
# and is the difference from mu + user_effect + movie_effect
# Compute rating from genre factor and find difference from mu + bu_1 + bi_1 (bg_1 = Y - mu + bu_1 + bi_1)
bg_1 <- train_edx %>% 
  left_join(bu_1, by = "userId") %>%
  left_join(bi_1, by = "movieId") %>%
  group_by(genres) %>%
  summarize(bg_1 = mean(rating - mu - bu_1 - bi_1))

# Predict BASELINE model rating from mu and initial USER, MOVIE, and GENRE effects
# Y = mu + bu + bi + bg
pred_bu_bi_bg_1 <- test_edx %>%
  left_join(bu_1, by = "userId") %>%
  left_join(bi_1, by = "movieId") %>%
  left_join(bg_1, by = "genres") %>% 
  mutate(pred = mu + bu_1 + bi_1 + bg_1) %>%
  pull(pred)

# And calculate BASELINE RMSE
baseline_rmse <- RMSE(test_edx$rating, pred_bu_bi_bg_1)

# Tabulate RMSEs so far:
model_comparison <- tibble(Model = c("Just the Average", 
                                     "User Effect", 
                                     "User + Movie Effects", 
                                     "Baseline Model - User, Movie, Genre"),
                           RMSE = c(base_0_rmse, base_1_RMSE, base_2_RMSE, baseline_rmse))
model_comparison

# The GENRE effect has an almost immeasurable impact on the baseline model and should be analyzed further in better format than initial group_by(genres) idea



## EXPERIMENT WITH LINEAR MODEL for capturing GENRE effect

# Add object to store genre names
all_genres <- unique(unlist(str_split(edx$genres,"\\|"))) [-20]

# PRE-PROCESSING: Add expanded genres to edx data set, one-hot encoding
edx_temp <- edx %>%
  cbind(mtabulate(str_split(edx$genres, "\\|"))) %>%
  select(-genres, -"(no genres listed)")

# Partition edx into training and test sets
set.seed(10)
edx_test_index <- createDataPartition(y = edx_temp$rating, times = 1, p = 0.1, list = FALSE)
train_edx_genres <- edx_temp[-edx_test_index,]
temp3 <- edx_temp[edx_test_index,]

# Make sure userId and movieId in test set are also in train set to avoid NAs
test_edx_genres <- temp3 %>% 
  semi_join(train_edx_genres, by = "movieId") %>%
  semi_join(train_edx_genres, by = "userId")

# Add rows removed from test set back into edx set
removed <- anti_join(temp3, test_edx_genres)
train_edx_genres <- rbind(train_edx_genres, removed)

# Clean up unused objects from environment
rm(edx_temp, edx_test_index, temp3, removed)

# Save train and test sets as objects
save(train_edx_genres, file = "rda/train_edx_genres.RData")
save(test_edx_genres, file = "rda/test_edx_genres.RData")

# Train linear model for genre predictor
bg_lm <- train_edx_genres %>% select(rating, all_of(all_genres)) %>% lm(rating ~ ., data = .)

# Compute rating from genre factor and find difference from mu
temp_bg_2 <- test_edx_genres %>%
  left_join(bu_1, by = "userId") %>%
  left_join(bi_1, by = "movieId")

bg_2 <- temp_bg_2 %>% mutate(bg_2 = mu - predict(bg_lm, newdata = temp_bg_2)) 

# UPDATED BASELINE model rating from mu and initial USER, MOVIE, and expanded GENRE effects
# Y = mu + bu + bi + bg_lm

pred_bu_bi_bg_lm <- bg_2 %>% 
  mutate(pred = mu + bu_1 + bi_1 + bg_2) %>%
  pull(pred)

# And calculate BASELINE RMSE
baseline2_rmse <- RMSE(pred_bu_bi_bg_lm, test_edx$rating)

# Tabulate RMSEs so far:
model_comparison <- tibble(Model = c("Just the Average", 
                                     "User Effect", 
                                     "Baseline Model - User + Movie Effects",
                                     "Baseline + Genre"),
                           RMSE = c(base_0_rmse, base_1_RMSE, base_2_RMSE, baseline2_rmse))
model_comparison



## REGULARIZATION OF USER AND MOVIE EFFECTS

# REGULARIZE user and movie predictors, use penalized least squares to shrink large error/variability due to small number of ratings
# Set overall average rating
overall <- mean(train_edx$rating)

# Regularization function for lambda
regulizer <- function(lambda, training, testing){
  # Define predictors:
  bu <- training %>% 
    group_by(userId) %>% 
    summarize(bu = sum(rating - overall)/(n() + lambda))
  bi <- training %>% 
    left_join(bu, by = "userId") %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - bu - overall)/(n() + lambda))
  # Predict ratings:
  preds <- testing %>%
    left_join(bu, by = "userId") %>%
    left_join(bi, by = "movieId") %>% 
    mutate(pred = overall + bu + bi) %>%
    pull(pred)
  return(RMSE(preds, testing$rating))
}

# Tune lambda to yield best RMSE
lambdas <- seq(0, 50, 1)

lamb_results <- sapply(lambdas, regulizer,
                       training = train_edx,
                       testing = test_edx)

# Plot results
lambda_tune <- tibble(Lambda = lambdas, RMSE = lamb_results) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point()
lambda <- lambdas[which.min(lamb_results)]

# Update baseline model with regularized USER and MOVIE effect, leave out GENRE negligible impact
bu <- train_edx %>% 
  group_by(userId) %>% 
  summarize(bu = sum(rating - overall)/(n() + lambda))
bi <- train_edx %>% 
  left_join(bu, by = "userId") %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - bu - overall)/(n() + lambda))
# Predict ratings:
pred_bu_bi_reg <- test_edx %>%
  left_join(bu, by = "userId") %>%
  left_join(bi, by = "movieId") %>% 
  mutate(pred = overall + bu + bi) %>%
  pull(pred)
baseline_reg_rmse <- RMSE(test_edx$rating, pred_bu_bi_reg)

# Update model_comparison table
model_comparison <- tibble(Model = c("Just the Average", 
                                     "User Effect", 
                                     "Baseline Model - User + Movie Effects",
                                     "Baseline + Genre",
                                     "Baseline - Regularized User and Movie"),
                           RMSE = c(base_0_rmse, base_1_RMSE, base_2_RMSE, baseline2_rmse, baseline_reg_rmse))
model_comparison



## MATRIX FACTORIZATION MODEL

# Use recosystem package to generate a matrix factorization model using the LIBMF library
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

# Update model_comparison table
model_comparison <- tibble(Model = c("Just the Average", 
                                     "User Effect", 
                                     "Baseline Model - User + Movie Effects",
                                     "Baseline + Genre",
                                     "Regularized Baseline - User and Movie",
                                     "Matrix Factorization - User + Movie"),
                           RMSE = c(base_0_rmse, base_1_RMSE, base_2_RMSE, baseline2_rmse, baseline_reg_rmse, matrix_fact_rmse))
model_comparison


## TEST MODEL ON FINAL_HOLDOUT_TEST ##
load("./rda/final_holdout_test.RData")

# Predict ratings on final holdout with Regularized Baseline linear model:
pred_final_reg <- final_holdout_test %>%
  left_join(bu, by = "userId") %>%
  left_join(bi, by = "movieId") %>% 
  mutate(pred = overall + bu + bi) %>%
  pull(pred)
final_reg_rmse <- RMSE(final_holdout_test$rating, pred_final_reg)

# Predict ratings on final holdout with Matrix Factorization model:
final_rec <- with(final_holdout_test, data_memory(user_index = userId,
                                       item_index = movieId,
                                       rating = rating))
pred_final_matrix <- rec$predict(final_rec, out_memory())
matrix_final_rmse <- RMSE(final_holdout_test$rating, pred_final_matrix)

# Final model comparison
final_model_comparison <- tibble(Model = c("Just the Average", 
                                     "User Effect", 
                                     "Baseline Model - User + Movie Effects",
                                     "Baseline + Genre",
                                     "Regularized Baseline - User and Movie",
                                     "Matrix Factorization - User + Movie",
                                     "Final Regularized Baseline",
                                     "Final Matrix Factorization"),
                           RMSE = c(base_0_rmse, 
                                    base_1_RMSE, 
                                    base_2_RMSE, 
                                    baseline2_rmse, 
                                    baseline_reg_rmse, 
                                    matrix_fact_rmse,
                                    final_reg_rmse,
                                    matrix_final_rmse))
final_model_comparison
