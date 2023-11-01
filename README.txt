The MovieLens Final Project creates a movie recommendation system from a 10 million observations data set adapted from the Netflix Prize challenge. 



SCRIPTS / FILES

MovieLens_Instructions - project guidelines from course, with grading rubric

partition_test_train_sets.R - creates the 'edx' (training set) and final_holdout_test (test set) data sets for training and testing machine learning algorithms used in the recommendation system, saves files to rda directory
  - edx.rda
  - final_holdout_test.rda

data-exploration.R - commented code for exploring and viewing data

genres_processing.R - script for string splitting genres into one hot encoding format and linear model for genres effect

matrix_factorization.R - use recosystem to write matrix factorization model for user and movie effects

pre-quiz.R - answers to questions from the MovieLens pre-project quiz

regularize_user_movie.R - regularization and lambda tuning for user and movie effects



DATA

edx.RData - clean data set from movielens import

final_holdout_test.RData - validation test set for calculation of final RMSEs

test_edx_genres.RData - testing data set partitioned from edx.RData with one hot encoding for 19 genres

train_edx_genres.RData - training data set partitioned from edx.RData with one hot encoding for genres



DRAFT REPORT

Contains the intermediate files and scripts generated during writing of the final report.



FINAL SUBMISSION

movielens-report-final-rydob2k.Rmd - Final project report in Rmarkdown for submission and grading

movielens-report-final-rydob2k.pdf - Final project report in Rmarkdown for submission and grading

movielens-script-final-rydob2k.R - Final project script containing code for data import, wrangling, model generation and evaluation, and RMSE calculation of final model