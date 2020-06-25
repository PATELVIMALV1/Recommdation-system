 install.packages("recommenderlab")
library(recommenderlab)
library(reshape2)


####### Example: Data generated in class #####

recommend_list <- read.csv(choose.files())
head(recommend_list)

View(recommend_list)
dim(recommend_list)

str(recommend_list)

###rating distribution######
hist(recommend$ratings)

#the datatype should be realRatingMatrix inorder to build recommendation engine
book_data_matrix <- as(recommend, 'realRatingMatrix')

#Popularity based 

book_recomm_model1 <- Recommender(book_data_matrix, method="POPULAR")


#Predictions for user 
recommended_book1 <- predict(book_recomm_model1, book_data_matrix[400:401], n=1)
as(recommended_book1, "list")


#User Based Collaborative Filtering

book_recomm_model2 <- Recommender(book_data_matrix, method="UBCF")

#Predictions for two users 
recommended_items2 <- predict(book_recomm_model2, book_data_matrix[413:414], n=5)
as(recommended_items2, "list")




