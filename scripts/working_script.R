# Load packages:
library(tidyverse)
if(!require(caret)) {install.packages("caret"); library(caret)}
if(!require(qdapTools)) {install.packages("qdapTools"); library(qdapTools)}

# Optional for multi-core processing
if(!require(doMC)) {install.packages("doMC"); library(doMC)} 
registerDoMC(cores = 10)

load("~/Projects/MovieLens/rda/edx.RData")


# Create 'RMSE' function for comparing model success
RMSE <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}


# Add movie release year (class = "Date") from title as variable in edx
  # Load lubridate package
  if(!require(lubridate)) {install.packages("lubridate"); library(lubridate)}

edx <- edx %>% mutate(release = ymd(as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\))")), truncated = 2L))

  # (test code)
  listy <- slice_sample(edx, n = 10)
  pattern <- "(?<=\\()\\d{4}(?=\\))"
  str_extract(listy, pattern)
  listy <- listy %>% mutate(release = as.integer(str_extract(title, "(?<=\\()\\d{4}(?=\\))")))
  rm(listy, pattern) #remove objects


# Partition edx into training and test sets
set.seed(10)
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_edx <- edx[-edx_test_index,]
temp <- edx[edx_test_index,]

# Make sure userId and movieId in test set are also in train set to avoid NAs
test_edx <- temp %>% 
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, test_edx)
train_edx <- rbind(train_edx, removed)

# Clean up unused object from environment
rm(edx_test_index, temp, removed)

## Construct BASELINE MODEL: rating = avg_rating + user_effect + movie_effect + genre_effect
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


## Improve Baseline Predictors

# Regularize USER and MOVIE effects
# We know from data exploration that the number of ratings each user submitted was highly variable and may impact that user's average rating
# So we regularize using penalized least squares for the number of ratings for each user: mean(y - mu - bu - bi)^2
