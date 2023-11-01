# EXPLORE AND DEFINE PREDICTORS TO BEAT THE BASELINE MODEL RMSE
  # Baseline model: Rating = USER + MOVIE + GENRE_effects
  # Y = bu_1 + bi_1 + bg_1
  # Baseline_RMSE = 0.880897

# LOAD packages:
library(tidyverse)
if(!require(caret)) {install.packages("caret"); library(caret)}
if(!require(qdapTools)) {install.packages("qdapTools"); library(qdapTools)}

if(!require(doMC)) {install.packages("doMC"); library(doMC)} # for multi-core processing
registerDoMC(cores = 10)

load("~/Projects/MovieLens/rda/edx.RData")


# RMSE function for comparing model success
RMSE <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}


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
tibble(Lambda = lambdas, RMSE = lamb_results) %>%
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
                                     "Baseline - User + Movie Effects",
                                     "Baseline - Regularized User and Movie"),
                           RMSE = c(base_0_rmse, base_1_RMSE, base_2_RMSE, baseline_reg_rmse))
model_comparison
