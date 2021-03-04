##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#title = as.character(title),
#genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Quiz
##########################################################

# shape of the data
dim(edx)

# number of 0 & 3 star ratings
edx %>% summarise(zeros=sum(rating==0), threes=sum(rating==3))

# number of unique films
edx %>% select(movieId) %>% unique() %>% count()

# number of unique users
n_distinct(edx$userId)

# number of movies in 4 popular genres
edx %>% summarise(Drama=sum(str_detect(genres, 'Drama')),
                  Comedy=sum(str_detect(genres, 'Comedy')),
                  Thriller=sum(str_detect(genres, 'Thriller')),
                  Romance=sum(str_detect(genres, 'Romance')))

# most rated movies
edx %>% group_by(title) %>% summarise(n=n()) %>% arrange(desc(n))

# rating frequency
edx %>% group_by(rating) %>% summarise(n=n()) %>% arrange(desc(n))

edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()

##########################################################
# Ray Exploration
##########################################################

head(edx)

# Subset the data to a manageable size (error when trying full dataset)
sm_index <- sample(1:nrow(edx), 1000, replace = F)
sm_movies <- edx[sm_index,]

# Create a User x Movie matrix
user_movie <- sm_movies %>% select(userId, movieId, rating) %>% 
  spread(movieId, rating) %>% as.matrix()
rownames(user_movie) <- user_movie[,1]
user_movie <- user_movie[,-1]
plot(c(user_movie)) # visualize the ratings
# Convert to residuals (remove column and row effects)
user_movie <- sweep(user_movie, 2, colMeans(user_movie, na.rm = T))
plot(c(user_movie)) # after centering movies
user_movie <- sweep(user_movie, 1, rowMeans(user_movie, na.rm = T))
plot(c(user_movie)) # after centering users

# Principal Component Analysis
user_movie[is.na(user_movie)] <- 0 #set na values to zero
pca <- prcomp(user_movie)
dim(pca$rotation) # q-vectors (principal components)
dim(pca$x) # p-vectors (user-effects)
##########################################################
# Data Wrangling
##########################################################
# Converts timestamp to use-able date of review format
library(lubridate)
edx <- mutate(edx, date = as_datetime(timestamp))

# Extracts release year
library(stringr)
pattern <- "\\(\\d{4}\\)"
pattern2 <- "\\d{4}"
edx <- mutate(edx, release=str_extract(edx$title, pattern)) %>%
  mutate(release=as.numeric(str_extract(edx$release, pattern2)))





##########################################################
# Disproved Hypothesis
##########################################################
# Find seasonal movies (Christmas/Holloween/ect)
seasonal <- edx %>% group_by(movieId) %>% 
  mutate(season = if(mean(month(date) == 01) >= 0.6){1}
         else if(mean(month(date) == 02) >= 0.9){2}
         else if(mean(month(date) == 03) >= 0.9){3}
         else if(mean(month(date) == 04) >= 0.9){4}
         else if(mean(month(date) == 05) >= 0.9){5}
         else if(mean(month(date) == 06) >= 0.9){6}
         else if(mean(month(date) == 07) >= 0.9){7}
         else if(mean(month(date) == 08) >= 0.9){8}
         else if(mean(month(date) == 09) >= 0.9){9}
         else if(mean(month(date) == 10) >= 0.9){10}
         else if(mean(month(date) == 11) >= 0.9){11}
         else if(mean(month(date) == 12) >= 0.9){12}
         else {0}
  )
seasonal %>% group_by(season) %>% summarise(n=n())
# December appears to have the strongest month-bias, however titles appear
# do not seem to correspond to holiday movies
seasonal[seasonal$season == 12,] %>% group_by(title) %>% summarise(n=n()) %>%
  arrange(desc(n))

# Christmas movies (proof that seasonal movie discrimination is pointless)
christmas_pattern <- "Christmas"
christmas_movies <- edx %>% 
  mutate(xmas = ifelse(str_detect(edx$title, christmas_pattern), 1, 0))
christmas_movies <- christmas_movies[christmas_movies$xmas==1] %>% group_by(title)

christmas_movies %>% ggplot(aes(month(date))) + geom_bar()







##########################################################
# Model Building / Select Machine Learning Algorithm
##########################################################

# Split the training set into training/test sets to design/test models
set.seed(420, sample.kind = 'Rounding')
test_index <- createDataPartition(y=edx$rating, times = 1, p=0.1, list=F)
edx_test <- edx[test_index,]
edx_train <- edx[-test_index,]






# This ensures we don't test on movies/users we have never seen before
edx_test <- edx_test %>% semi_join(edx_train, by='movieId') %>%
  semi_join(edx_train, by='userId')

# This model predicts the avg movie rating for all cases
mu <- edx_train$rating %>% mean()
(naive_rmse <- RMSE(edx_test$rating, mu))






# This model builds on previous by introducing a movie bias term
movie_avgs <- edx_train %>% group_by(movieId) %>% summarise(b_i=mean(rating-mu))
movie_bias_model <- mu + edx_test %>% left_join(movie_avgs, by='movieId') %>% pull(b_i)
(movie_bias_rmse <- RMSE(edx_test$rating, movie_bias_model))







# This model builds on previous by introducing a user bias term
user_avgs <- edx_train %>% left_join(movie_avgs, by='movieId') %>% group_by(userId) %>% summarise(b_u=mean(rating-mu-b_i))
user_movie_bias_model <- edx_test %>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>%
  mutate(pred=mu+b_i+b_u) %>% pull(pred)
(user_movie_bias_rmse <- RMSE(edx_test$rating, user_movie_bias_model))







# This model builds on previous by introducing a regularization (lambda) term 

# Use cross-validation to search for best lambda term:
#lambdas <- seq(0, 10, 0.25)
lambdas <- seq(4, 5, 0.1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_train$rating)
  b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i=sum(rating-mu)/(n()+l))
  
  b_u <- edx_train %>% left_join(b_i, by='movieId') %>%
    group_by(userId) %>% summarise(b_u=sum(rating-b_i-mu)/(n()+l))
  
  predicted_ratings <- edx_test %>% left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>% mutate(pred=mu+b_i+b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx_test$rating))
})
# visualize the search for best lambda
qplot(lambdas, rmses)
# save the best lambda term
(lambda <- lambdas[which.min(rmses)])

#Regularized movie bias term
b_i <- edx_train %>% group_by(movieId) %>% summarise(b_i=sum(rating-mu)/(n()+lambda))
#Regularized user bias term
b_u <- edx_train %>% left_join(b_i, by='movieId') %>%
  group_by(userId) %>% summarise(b_u=sum(rating-b_i-mu)/(n()+lambda))

regularized_user_movie_model <- edx_test %>% left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>% mutate(pred=mu+b_i+b_u) %>% pull(pred)
(regularized_user_movie_rmse <- RMSE(edx_test$rating, regularized_user_movie_model))







#This model will use matrix factorization 
#This is an alternative method that is not restricted by memory
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
set.seed(69, sample.kind = 'Rounding')

#inputs are stored in memory
train_set <- with(edx_train, data_memory(user_index=userId, item_index=movieId, rating=rating))
test_set <- with(edx_test, data_memory(user_index=userId, item_index=movieId, rating=rating))

r <- Reco()

#train with default parameters
r$train(train_set)

#predict
matrix_factor <- r$predict(test_set, out_memory())

(matrix_factor_rmse <- RMSE(matrix_factor, edx_test$rating))







# Evaluate the different models
methods <- c('Just the average', '+ movie bias', '+ user bias', 'regularized model', 'matrix factorization')
rmses <- c(naive_rmse, movie_bias_rmse, user_movie_bias_rmse, regularized_user_movie_rmse, matrix_factor_rmse)
(model_evaluations <- tibble(method=methods, RMSE=rmses))

##########################################################
# Final Model 
##########################################################