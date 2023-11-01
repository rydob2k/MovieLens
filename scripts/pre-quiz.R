# MovieLens Pre-Quiz
load("rda/edx.rda")
# Q5
  # Detect function:
count(edx, str_detect(edx$genres, "Drama"))
count(edx, str_detect(edx$genres, "Comedy"))
count(edx, str_detect(edx$genres, "Thriller"))
count(edx, str_detect(edx$genres, "Romance"))
  # Answer code: with str_detect
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# Q6 
edx %>% group_by(movieId, title) %>% summarize(ratings = n()) %>%
  slice_max(order_by = ratings, n = 5)
  # Answer code:
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Q7
edx |> group_by(rating) |> summarize(count = n()) |> arrange(desc(count))
  # Answer:
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))

# Q8 Answer:
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()
