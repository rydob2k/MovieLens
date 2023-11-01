library(tidyverse)
if(!require(caret)) {install.packages("caret"); library(caret)}
if(!require(qdapTools)) {install.packages("qdapTools"); library(qdapTools)}
load("~/Projects/MovieLens/rda/edx.RData")

# Exploratory leftovers
  movieId_avg_ratings <- edx %>%
    summarize(avg_rating = mean(rating), .by = movieId)
  movieId_genres <- edx %>% distinct(movieId, .keep_all = TRUE) %>%
    select(movieId, genres) %>%
    left_join(movieId_avg_ratings, by = "movieId")
  genres_matrix <- movieId_genres %>% 
    cbind(mtabulate(str_split(movieId_genres$genres, "\\|"))) %>%
    select(-genres, -"(no genres listed)")



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

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp3, test_edx_genres)
train_edx_genres <- rbind(train_edx_genres, removed)

# Clean up unused object from environment
rm(edx_temp, edx_test_index, temp3, removed)

# Save train and test sets as objects
save(train_edx_genres, file = "rda/train_edx_genres.RData")
save(test_edx_genres, file = "rda/test_edx_genres.RData")




  
  # Add object to store genre names
all_genres <- unique(unlist(str_split(edx$genres,"\\|"))) [-20]

# Add expanded genres to train_edx data set
train_edx_genres <- train_edx %>%
  cbind(mtabulate(str_split(train_edx$genres, "\\|"))) %>%
  select(-"(no genres listed)")

# Train linear model for genre predictor
bg_lm <- train_edx_genres %>% select(rating, all_of(all_genres)) %>% lm(rating ~ ., data = .)

# Compute rating from genre factor and find difference from mu
temp_bg_2 <- test_edx %>%
  left_join(bu_1, by = "userId") %>%
  left_join(bi_1, by = "movieId") %>%
  cbind(mtabulate(str_split(test_edx$genres, "\\|")))

bg_2 <- temp_bg_2 %>% 
  mutate(bg_2 = mu - predict(bg_lm, newdata = temp_bg_2)) 

# Predict updated BASELINE model rating from mu and initial USER, MOVIE, and expanded GENRE effects
# Y = mu + bu + bi + bg

pred_bu_bi_bg_2 <- bg_2 %>% 
  mutate(pred = mu + bu_1 + bi_1 + bg_2) %>%
  pull(pred)

# And calculate BASELINE RMSE
baseline2_rmse <- RMSE(pred_bu_bi_bg_2, test_edx$rating)

# Tabulate RMSEs so far:
model_comparison <- tibble(Model = c("Just the Average", 
                                     "User Effect", 
                                     "User + Movie Effects", 
                                     "Baseline Model - User, Movie, Genre",
                                     "Baseline + Genre"),
                           RMSE = c(base_0_rmse, base_1_RMSE, base_2_RMSE, baseline_rmse, baseline2_rmse))
model_comparison
