# Ray Sherbourne
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
# Data Wrangling & Exploration
##########################################################
head(edx)

# Converts timestamp to use-able date of review format
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(lubridate)
edx <- mutate(edx, date = as_datetime(timestamp))

# Extracts release year
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
library(stringr)
pattern <- "\\(\\d{4}\\)"
pattern2 <- "\\d{4}"
edx <- mutate(edx, release=str_extract(edx$title, pattern))#%>%
edx <- edx %>% mutate(release=as.numeric(str_extract(edx$release, pattern2)))

# Creates binary predictor for each genre
edx <- edx %>% mutate(Action = ifelse(str_detect(.$genres, 'Action'), 1, 0),
                      Adventure = ifelse(str_detect(.$genres, 'Adventure'), 1, 0),
                      Animation = ifelse(str_detect(.$genres, 'Animation'), 1, 0),
                      Children = ifelse(str_detect(.$genres, 'Children'), 1, 0),
                      Comedy = ifelse(str_detect(.$genres, 'Comedy'), 1, 0),
                      Crime = ifelse(str_detect(.$genres, 'Crime'), 1, 0),
                      Docu = ifelse(str_detect(.$genres, 'Documentary'), 1, 0),
                      Drama = ifelse(str_detect(.$genres, 'Drama'), 1, 0),
                      Fantasy = ifelse(str_detect(.$genres, 'Fantasy'), 1, 0),
                      Noir = ifelse(str_detect(.$genres, 'Noir'), 1, 0),
                      Horror = ifelse(str_detect(.$genres, 'Horror'), 1, 0),
                      Musical = ifelse(str_detect(.$genres, 'Musical'), 1, 0),
                      Mystery = ifelse(str_detect(.$genres, 'Mystery'), 1, 0),
                      Romance = ifelse(str_detect(.$genres, 'Romance'), 1, 0),
                      SciFi = ifelse(str_detect(.$genres, 'Sci'), 1, 0),
                      Thriller = ifelse(str_detect(.$genres, 'Thriller'), 1, 0),
                      War = ifelse(str_detect(.$genres, 'War'), 1, 0),
                      Western = ifelse(str_detect(.$genres, 'Western'), 1, 0))

# Visualize the predictors

# User ratings
edx %>% group_by(userId) %>% summarise(n = n()) %>% 
  ggplot(aes(n)) + geom_histogram() + scale_x_continuous(trans='log10')+
  ggtitle('Users')+ xlab('Number of Movie Ratings') + ylab('Number of Users')
# Movie ratings
edx %>% group_by(movieId) %>% summarise(n = n()) %>% 
  ggplot(aes(n)) + geom_histogram() + scale_x_continuous(trans='log10')+
  ggtitle('Ratings')+xlab('Number of Movie Ratings')+ylab('Number of Movies')
# Ratings
edx %>% group_by(rating) %>% summarise(n = n()) %>%  
  ggplot(aes(x=rating, y=n)) + geom_line() +
  ggtitle('Movie Ratings') + ylab('Number of Ratings')
# Timestamp
edx %>% mutate(date=round_date(date, unit='week')) %>% group_by(date) %>%
  summarise(rating=mean(rating)) %>%
  ggplot(aes(x=date, y=rating)) + geom_point() + geom_smooth()+
  ggtitle('Average Rating per week') + ylab('Rating')
# Genres
(genre_count <-edx %>% distinct(edx$movieId, .keep_all = TRUE) %>% select(9:26) %>% 
    colSums()) %>% barplot(las=2, main = 'Movie Genres', ylab = 'Count')

# Christmas movies (proof that holiday movie discrimination is pointless)
christmas_pattern <- "Christmas"
christmas_movies <- edx %>% 
  mutate(xmas = ifelse(str_detect(edx$title, christmas_pattern), 1, 0))
christmas_movies <- christmas_movies[christmas_movies$xmas==1] %>% group_by(title)

christmas_movies %>% ggplot(aes(x=month(date))) + geom_bar(stat = 'count') +
  scale_x_continuous(breaks=c(1:12)) +
  ggtitle("Reviews for Christmas Movies")+
  xlab('Month')



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
rm(test_index)

# This model predicts the avg movie rating for all cases
mu <- edx_train$rating %>% mean()
naive_rmse <- RMSE(edx_test$rating, mu)

methods <- ('Just the average')
rmses <- (naive_rmse)
(model_evaluations <- tibble(method=methods, RMSE=rmses))

# This model builds on previous by introducing a movie bias term
movie_avgs <- edx_train %>% group_by(movieId) %>% summarise(b_i=mean(rating-mu))
movie_bias_model <- mu + edx_test %>% left_join(movie_avgs, by='movieId') %>% pull(b_i)
movie_bias_rmse <- RMSE(edx_test$rating, movie_bias_model)

methods <- c('Just the average', '+ movie bias')
rmses <- c(naive_rmse, movie_bias_rmse)
(model_evaluations <- tibble(method=methods, RMSE=rmses))

# This model builds on previous by introducing a user bias term
user_avgs <- edx_train %>% left_join(movie_avgs, by='movieId') %>% group_by(userId) %>% summarise(b_u=mean(rating-mu-b_i))
user_movie_bias_model <- edx_test %>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>%
  mutate(pred=mu+b_i+b_u) %>% pull(pred)
user_movie_bias_rmse <- RMSE(edx_test$rating, user_movie_bias_model)

methods <- c('Just the average', '+ movie bias', '+ user bias')
rmses <- c(naive_rmse, movie_bias_rmse, user_movie_bias_rmse)
(model_evaluations <- tibble(method=methods, RMSE=rmses))

################################################
# Reversed Bias Terms
###############################################
# This model builds on mu by introducing a user bias term
user_avgs <- edx_train %>% group_by(userId) %>% summarise(b_u=mean(rating-mu))
user_bias_model <- mu + edx_test %>% left_join(user_avgs, by='userId') %>% pull(b_u)
user_bias_rmse <- RMSE(edx_test$rating, user_bias_model)

# This model builds on previous by introducing a movie bias term
movie_avgs <- edx_train %>% left_join(user_avgs, by='userId') %>% group_by(movieId) %>% summarise(b_i=mean(rating-mu-b_u))
movie_user_bias_model <- edx_test %>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>%
  mutate(pred=mu+b_i+b_u) %>% pull(pred)
movie_user_bias_rmse <- RMSE(edx_test$rating, movie_user_bias_model)

methods <- c('Just the average', '+ movie bias', '+ user bias', 'reversed_biases')
rmses <- c(naive_rmse, movie_bias_rmse, user_movie_bias_rmse, movie_user_bias_rmse)
(model_evaluations <- tibble(method=methods, RMSE=rmses))
################################################
# Reversed Bias Terms
###############################################

# This model builds on biased model by introducing a regularization (lambda) term 

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
regularized_user_movie_rmse <- RMSE(edx_test$rating, regularized_user_movie_model)

methods <- c('Just the average', '+ movie bias', '+ user bias', 'reversed_biases', 'regularized model')
rmses <- c(naive_rmse, movie_bias_rmse, user_movie_bias_rmse, movie_user_bias_rmse, regularized_user_movie_rmse)
(model_evaluations <- tibble(method=methods, RMSE=rmses))


#This model will use matrix factorization without tuning parameters
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
set.seed(69, sample.kind = 'Rounding')

train_set <- with(edx_train, data_memory(user_index=userId, item_index=movieId, rating=rating))
test_set <- with(edx_test, data_memory(user_index=userId, item_index=movieId, rating=rating))

# Create a model object
r <- Reco()

#train with tuned parameters
r$train(train_set)

#predict
matrix_factor <- r$predict(test_set, out_memory())

matrix_factor_rmse <- RMSE(matrix_factor, edx_test$rating)


#This model will tune the training parameters

# Call the $tune() method to select the best parameters with cross validation 
# (This will take a while, see progress bar in R console)
opts <- r$tune(train_set, opts = list(dim=c(10, 20, 30), #dim is number of latent factors
                                      lrate=c(0.1, 0.2), #learning rate (step size in gradient decent)
                                      costp_l1=0, costq_l1=0, #L1 regularization terms set to 0 (default is c(0.01, 0.1))
                                      nthread=1, #number of threads for parallel computing
                                      niter=10)) #number of iterations


#train with tuned parameters
r$train(train_set, opts=c(opts$min, nthread=1, niter=20))

#predict
matrix_factor_tuned <- r$predict(test_set, out_memory())

matrix_factor_tuned_rmse <- RMSE(matrix_factor_tuned, edx_test$rating)

# Evaluate the different models
methods <- c('Just the average', '+ movie bias', '+ user bias', 'reversed_biases', 'regularized model', 'matrix factorization', 'matrix factorization (Tuned)')
rmses <- c(naive_rmse, movie_bias_rmse, user_movie_bias_rmse, movie_user_bias_rmse, regularized_user_movie_rmse, matrix_factor_rmse, matrix_factor_tuned_rmse)
(model_evaluations <- tibble(method=methods, RMSE=rmses))

##########################################################
# Final Model 
##########################################################

#This model will use matrix factorization without tuning parameters
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
set.seed(69, sample.kind = 'Rounding')

train_set <- with(edx, data_memory(user_index=userId, item_index=movieId, rating=rating))
test_set <- with(validation, data_memory(user_index=userId, item_index=movieId, rating=rating))

# Create a model object
r <- Reco()

# Tune parameters with cross validation
opts <- r$tune(train_set, opts = list(dim=c(10, 20, 30), #dim is number of latent factors
                                      lrate=c(0.1, 0.2), #learning rate (step size in gradient decent)
                                      costp_l1=0, costq_l1=0, #L1 regularization terms set to 0 (default is c(0.01, 0.1))
                                      nthread=1, #number of threads for parallel computing
                                      niter=10)) #number of iterations
#train with tuned parameters
r$train(train_set, opts=c(opts$min, nthread=1, niter=20))

#predict
final_prediction <- r$predict(test_set, out_memory())

(final_rmse <- RMSE(final_prediction, validation$rating))
