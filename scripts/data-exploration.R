library(tidyverse)
library(caret)
library(doMC)
registerDoMC(cores = 10)


## Explore Problem

# How to predict what user will rate a movie?
  # Interaction between a user and a movie can be modeled based on that user's data and that movie's data.
  # Factors that influence/bias: genre, movie popularity, actors, time, other users' ratings?

load("~/Projects/MovieLens/rda/edx.RData")
load("~/Projects/MovieLens/rda/final_holdout_test.RData")

# Overview of Training Data - MovieLens 10M
tibble(edx) # table with variables
str(edx)
edx %>% group_by(userId) %>% summarize(n()) %>% nrow() # 69878 users
edx %>% group_by(movieId) %>% summarize(n()) %>% nrow() # 10677 movies
edx %>% summarize(Users = n_distinct(userId),
                   Movies = n_distinct(movieId))

# Not every user has rated every movie, ideally the recommendation model should predict what the rating would be.


## BASELINE MODEL: rating = avg_rating + user_effect + movie_effect + genre_effect
  # Y = mu + b_u + b_i + g 

# Create 'RMSE' function for comparing model success
RMSE <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}

# So for baseline model:
mu <- mean(train_edx$rating)

# and RMSE = 1.059813
baseline_rmse <- RMSE(test_edx$rating, mu)


## Explore user-effect and possible user related predictors

# Ratings overview by user
user_ratings <- edx %>% group_by(userId) %>% 
  summarize(n_ratings = n(), median_rating = median(rating), avg_rating = round(mean(rating), digits = 1))

# Skewed distribution of n_ratings data: half of users rated less than 62 movies but mean is 129 movies
user_ratings %>% summarise(average = mean(n_ratings), median = median(n_ratings), min = min(n_ratings), max = max(n_ratings))
user_ratings %>% filter(n_ratings <= (0.10)*max(n_ratings)) %>% 
  ggplot(aes(n_ratings)) + geom_histogram(binwidth = 10)

# Proportion users rating less movies than mean = 129, proportion = 0.72867, suggests potential effect on model
sum(user_ratings$n_ratings <= 129) / length(user_ratings$n_ratings)

# Large variation in avg_rating suggests potential effect on model
user_ratings %>% filter(n_ratings >= 30) %>% ggplot(aes(avg_rating)) + geom_histogram(bins = 20)



## Explore movie-effect and possible movie related predictors

# Ratings overview by movie
movie_ratings <- edx %>% group_by(movieId) %>% 
  summarize(title = title[1], i_ratings = n(), avg_i_rating = round(mean(rating), digits = 1))

# Highest rated movies are obscure titles with less than 5 reviews, so number of ratings affects avg rating
slice_max(movie_ratings, order_by = avg_i_rating, n = 10)
# Only 6% movies were rated greater than 4 stars
sum(movie_ratings$avg_i_rating >= 4) / length(movie_ratings$avg_i_rating)
sum(movie_ratings$avg_i_rating <= 2) / length(movie_ratings$avg_i_rating)
histogram(movie_ratings$avg_i_rating)


set.seed(10)
small_train_edx <- train_edx %>% group_by(movieId) %>% slice_sample(prop = 0.1)
fit_loess_small <- train(rating ~ movieId, data = small_train_edx, method = "gamLoess")

control <- trainControl(method = "cv", number = 10, p = .9)
fit_knn_small <- train(rating ~ movieId, data = small_train_edx, method = "knn", tuneGrid = data.frame(k = c(3,5,7)), trControl = control)
