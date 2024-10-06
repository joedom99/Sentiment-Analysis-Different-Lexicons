# Basic sentiment analysis of different lexicons
# See blog post on https://blog.marketingdatascience.ai
# Version 3 - Joe Domaleski

# Load necessary libraries
library(tidytext)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(vader)
library(ggplot2)

# Setup environment
rm(list = ls())
set.seed(1)

# Step 1: Define sentiment analysis functions

# Function for VADER sentiment analysis with progress printing
vader_sentiment_sequential <- function(reviews) {
  sapply(seq_along(reviews), function(i) {
    cat("Processing review", i, "out of", length(reviews), "\n")
    vader::vader_df(reviews[i])$compound
  })
}

# Function for sentiment analysis using specified lexicon
perform_sentiment_analysis <- function(data, lexicon = "bing") {
  tokenized_data <- data %>% unnest_tokens(word, review)
  
  if (lexicon == "afinn") {
    sentiment_data <- tokenized_data %>%
      distinct(word, .keep_all = TRUE) %>%
      inner_join(get_sentiments("afinn"), by = "word") %>%
      group_by(id) %>%
      summarise(sentiment_score = sum(value)) %>%
      right_join(data, by = "id") %>%
      mutate(sentiment_rating = case_when(
        sentiment_score > 0 ~ "positive",
        sentiment_score < 0 ~ "negative",
        TRUE ~ "neutral"
      ))
  } else if (lexicon == "vader") {
    vader_scores <- vader_sentiment_sequential(data$review)
    sentiment_data <- data %>%
      mutate(sentiment_score = unlist(vader_scores),
             sentiment_rating = case_when(
               sentiment_score > 0 ~ "positive",
               sentiment_score < 0 ~ "negative",
               TRUE ~ "neutral"
             ))
  } else {
    sentiment_data <- tokenized_data %>%
      distinct(word, .keep_all = TRUE) %>%
      inner_join(get_sentiments(lexicon), by = "word") %>%
      count(id, sentiment) %>%
      pivot_wider(names_from = sentiment, values_from = n, values_fill = list(n = 0)) %>%
      mutate(sentiment_score = positive - negative) %>%
      right_join(data, by = "id") %>%
      mutate(
        sentiment_rating = case_when(
          sentiment_score > 0 ~ "positive",
          sentiment_score < 0 ~ "negative",
          TRUE ~ "neutral"
        )
      )
  }
  
  return(sentiment_data)
}

# Step 2: Preparing the Data
reviews_data <- read_csv("yelp_reviews_bouchon_bakery_las_vegas.csv")

clean_reviews <- reviews_data %>%
  filter(!is.na(review)) %>%
  mutate(rating = ifelse(rating == "", NA, rating), id = row_number()) %>%
  unnest_tokens(word, review) %>%
  anti_join(stop_words, by = "word") %>%
  group_by(id) %>%
  summarise(review = paste(word, collapse = " "), rating = first(rating)) %>%
  ungroup()

# Step 3: Perform sentiment analysis using different lexicons with proper renaming
sentiment_bing <- perform_sentiment_analysis(clean_reviews, lexicon = "bing") %>%
  rename(sentiment_score_bing = sentiment_score)

sentiment_afinn <- perform_sentiment_analysis(clean_reviews, lexicon = "afinn") %>%
  rename(sentiment_score_afinn = sentiment_score)

sentiment_nrc <- perform_sentiment_analysis(clean_reviews, lexicon = "nrc") %>%
  rename(sentiment_score_nrc = sentiment_score)

sentiment_vader <- perform_sentiment_analysis(clean_reviews, lexicon = "vader") %>%
  rename(sentiment_score_vader = sentiment_score)

# Step 4: Combine sentiment scores from all lexicons
comparison_lexicons <- sentiment_bing %>%
  left_join(sentiment_afinn, by = c("id", "review")) %>%
  left_join(sentiment_nrc, by = c("id", "review")) %>%
  left_join(sentiment_vader, by = c("id", "review"))

# Step 5: Replace non-finite values with 0 and scale sentiment scores using min-max scaling
scale_scores <- function(scores) {
  (scores - min(scores, na.rm = TRUE)) / (max(scores, na.rm = TRUE) - min(scores, na.rm = TRUE))
}

comparison_scaled <- comparison_lexicons %>%
  mutate(
    sentiment_score_bing = ifelse(is.finite(sentiment_score_bing), sentiment_score_bing, 0),
    sentiment_score_afinn = ifelse(is.finite(sentiment_score_afinn), sentiment_score_afinn, 0),
    sentiment_score_nrc = ifelse(is.finite(sentiment_score_nrc), sentiment_score_nrc, 0),
    sentiment_score_vader = ifelse(is.finite(sentiment_score_vader), sentiment_score_vader, 0)
  ) %>%
  mutate(
    scaled_bing = scale_scores(sentiment_score_bing),
    scaled_afinn = scale_scores(sentiment_score_afinn),
    scaled_nrc = scale_scores(sentiment_score_nrc),
    scaled_vader = scale_scores(sentiment_score_vader)
  )

# Step 6: Plot scaled sentiment scores
# Sample the first 25 reviews for clearer plots
comparison_sampled <- comparison_scaled %>% slice(1:25)
comparison_sampled_long <- comparison_sampled %>%
  pivot_longer(cols = c(scaled_bing, scaled_afinn, scaled_nrc, scaled_vader),
               names_to = "lexicon", values_to = "scaled_score")

# Line plot for the first 25 reviews
ggplot(comparison_sampled_long, aes(x = id, y = scaled_score, color = lexicon)) +
  geom_line() +
  labs(title = "Scaled Sentiment Scores for First 25 Reviews",
       x = "Review ID", y = "Scaled Sentiment Score (0 to 1)") +
  theme_minimal()

# Box plot for all lexicons
ggplot(comparison_sampled_long, aes(x = lexicon, y = scaled_score, fill = lexicon)) +
  geom_boxplot() +
  labs(title = "Distribution of Scaled Sentiment Scores Across Lexicons",
       x = "Lexicon", y = "Scaled Sentiment Score (0 to 1)") +
  theme_minimal()

# Step 7: Z-score normalization of sentiment scores
z_score_normalization <- function(scores) {
  (scores - mean(scores, na.rm = TRUE)) / sd(scores, na.rm = TRUE)
}

comparison_z_scaled <- comparison_lexicons %>%
  mutate(
    sentiment_score_bing = ifelse(is.finite(sentiment_score_bing), sentiment_score_bing, 0),
    sentiment_score_afinn = ifelse(is.finite(sentiment_score_afinn), sentiment_score_afinn, 0),
    sentiment_score_nrc = ifelse(is.finite(sentiment_score_nrc), sentiment_score_nrc, 0),
    sentiment_score_vader = ifelse(is.finite(sentiment_score_vader), sentiment_score_vader, 0)
  ) %>%
  mutate(
    z_scaled_bing = z_score_normalization(sentiment_score_bing),
    z_scaled_afinn = z_score_normalization(sentiment_score_afinn),
    z_scaled_nrc = z_score_normalization(sentiment_score_nrc),
    z_scaled_vader = z_score_normalization(sentiment_score_vader)
  )

comparison_z_long <- comparison_z_scaled %>%
  pivot_longer(cols = c(z_scaled_bing, z_scaled_afinn, z_scaled_nrc, z_scaled_vader),
               names_to = "lexicon", values_to = "z_scaled_score")

# Z-score density plot

# Filter the data to include only Z-scores between -5 and 5
comparison_z_filtered <- comparison_z_long %>%
  filter(z_scaled_score >= -5 & z_scaled_score <= 5)

# Z-score density plot with filtered data
ggplot(comparison_z_filtered, aes(x = z_scaled_score, color = lexicon)) +
  geom_density() +
  labs(title = "Z-Score Normalized Sentiment Scores Across Lexicons",
       x = "Z-Scaled Sentiment Score", y = "Density") +
  theme_minimal()

