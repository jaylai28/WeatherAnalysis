##################### Library
library(tree)
library(e1071)
library(ROCR)
library(randomForest)
library(adabag)
library(rpart)
library(pastecs)
library(caret)
library(dplyr)
library(corrplot)

#########################
#Q1
#create our individual data and tidy the data by removing unnecessary data. rm(list = ls())
rm(list = ls())
waus = read.csv("WAUS2020.csv")
L = as.data.frame(c(1:49))
set.seed(29634431) # Student ID is the random seed
L = L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
waus = waus[(waus$Location %in% L),]
waus = waus[sample(nrow(waus), 2000, replace = FALSE),] # sample 2000 rows


#Format all the values in the dataset into 4 decimals
options(digits=4)

#get the summary of the waus dataset
summary(waus)
#look at the type of each variable in the dataset
str(waus)

#count the number of times it rains and does not rain over the year
x = waus %>% count(RainToday)
proportion = x$n[2]/x$n[1]
proportion

#get all the real-valued attributes and remove the NA values
real_value_waus = subset(waus, select = -c(WindGustDir,WindDir9am,WindDir3pm,RainToday,RainTomorrow))
real_value_waus = na.omit(real_value_waus)

#plot a heatmap to see the correlation of all the variables
factor_vars <- names(which(sapply(real_value_waus, class) == "factor"))
numeric_vars <- setdiff(colnames(real_value_waus), factor_vars)
numeric_vars <- setdiff(numeric_vars, "RainTomorrow")
numeric_vars
numeric_vars_mat <- as.matrix(real_value_waus[, numeric_vars, drop=FALSE])
numeric_vars_cor <- cor(numeric_vars_mat)
corrplot(numeric_vars_cor)


#omit the Date such as Day, Month, Year from the dataset
waus = subset(waus, select = -c(Day,Month,Year))


#Factor the categorical vaariables
waus$WindGustDir = factor(waus$WindDir9am)
waus$WindDir9am = factor(waus$WindDir9am)
waus$WindDir3pm = factor(waus$WindDir3pm)
waus$RainToday = factor(waus$RainToday)
waus$RainTomorrow = factor(waus$RainTomorrow)

#remove the NA values from the dataset
waus = na.omit(waus)

#get the description of the real-valued attributes after removing the NA values.
real_value_waus2 = subset(waus, select = -c(Location,WindGustDir,WindDir9am,WindDir3pm,RainToday,RainTomorrow))
stat.desc(real_value_waus2)

#count the number of times it rains and does not rain over the year
x = waus %>% count(RainToday)
proportion = x$n[2]/x$n[1]
proportion


#Q3
#Divide the data into 70% training set and 30% test set
set.seed(29634431) #Student ID as random seed 
train.row = sample(1:nrow(waus), 0.7*nrow(waus))
waus.train = waus[train.row,]
waus.test = waus[-train.row,]


#Q4
#Decision tree
waus_decisiontree = tree(RainTomorrow ~., data = waus.train)
summary(waus_decisiontree)

#Naive Bayes
waus_naivebayes = naiveBayes(RainTomorrow~., data = waus.train)

#Bagging
waus_bagging = bagging(RainTomorrow ~. , data = waus.train, mfinal=5)

#Boosting
waus_boosting = boosting(RainTomorrow ~. , data = waus.train, mfinal=10)

#Random Forest
waus_randomforest = randomForest(RainTomorrow ~. , data = waus.train, na.action = na.exclude)



#Q5
#Decision tree
#Calculate the accuracy for Decision Tree
waus_pred_decisiontree = predict(waus_decisiontree, waus.test, type = "class")
waus_pred_decisiontree
#Create confusion matrix
tree_confusion = table(predicted = waus_pred_decisiontree,observed = waus.test$RainTomorrow)
tree_confusion

#calculate the accuracy of decision tree with confusion matrix
table1 = confusionMatrix(waus_pred_decisiontree,waus.test$RainTomorrow)
table1
confusion_value1 = table1$overall['Accuracy']
confusion_value1

#Naive Bayes
#Calculate the accuracy for Naive Bayes
waus_pred_naivebayes = predict(waus_naivebayes, waus.test)

#Create confusion matrix
naivebayes_confusion=table(predicted = waus_pred_naivebayes, actual = waus.test$RainTomorrow)
naivebayes_confusion

#calculate the accuracy of decision tree with confusion matrix
table2 = confusionMatrix(waus_pred_naivebayes,waus.test$RainTomorrow)
confusion_value2 = table1$overall['Accuracy']
confusion_value2



#Bagging
#Calculate the accuracy for Bagging
waus_pred_bagging = predict.bagging(waus_bagging, waus.test)

#Create confusion matrix
waus_pred_bagging$confusion
confusion_value3 = (waus_pred_bagging$confusion[1,1]+waus_pred_bagging$confusion[2,2])/
  sum(waus_pred_bagging$confusion)
confusion_value3

#Boosting
#Calculate the accuracy for Boosting
waus_pred_boosting = predict.boosting(waus_boosting, newdata=waus.test)
waus_pred_boosting$confusion

confusion_value4 = (waus_pred_boosting$confusion[1,1]+waus_pred_boosting$confusion[2,2])/
  sum(waus_pred_boosting$confusion)
confusion_value4


#Random Forest
#Calculate the accuracy for Random Forest
waus_pred_randomforest = predict(waus_randomforest, waus.test)

#Create confusion matrix
randomforest_confusion = table(Predicted = waus_pred_randomforest, Actual = waus.test$RainTomorrow)
randomforest_confusion
#calculate the accuracy of decision tree with confusion matrix
table5 = confusionMatrix(waus_pred_randomforest,waus.test$RainTomorrow)
confusion_value5 = table5$overall['Accuracy']
confusion_value5


#Q6
#Decision Tree
# do predictions as probabilities and draw ROC
waus_pred_DT_test = predict(waus_decisiontree, waus.test, type = "vector")

# computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
# labels are actual values, predictors are probability of RainTomorrow
waus_pred_DT = prediction( waus_pred_DT_test[,2], waus.test$RainTomorrow)
waus_perf_DT = performance(waus_pred_DT,"tpr","fpr")
plot(waus_perf_DT, col='orange')
abline(0,1)

#Computing AUC for Decision Tree
waus_AUC_DT = performance(waus_pred_DT, "auc")
value1 = as.numeric(waus_AUC_DT@y.values)
value1


#Naïve Bayes
# Outputs as confidence levels and construct ROC curve for Naive Bayes
waus_pred_NB_test = predict(waus_naivebayes, waus.test, type = 'raw')
waus_pred_NB = prediction( waus_pred_NB_test[,2], waus.test$RainTomorrow)
waus_perf_NB = performance(waus_pred_NB,"tpr","fpr")
plot(waus_perf_NB, col = "blue")

#Computing AUC for Naive Bayes
waus_AUC_NB = performance(waus_pred_NB, "auc")
value2 = as.numeric(waus_AUC_NB@y.values)
value2

#Bagging
waus_pred_bagging
waus_pred_bag = prediction( waus_pred_bagging$prob[,2], waus.test$RainTomorrow)
waus_perf_bag = performance(waus_pred_bag,"tpr","fpr")
plot(waus_perf_bag, add=TRUE, col = "violet")

#Computing AUC for Bagging
waus_AUC_BG = performance(waus_pred_bag, "auc")
value3 = as.numeric(waus_AUC_BG@y.values)
value3

#Boosting
waus_pred_BT_test = predict.boosting(waus_boosting, newdata=waus.test)
waus_pred_BT = prediction( waus_pred_BT_test$prob[,2], waus.test$RainTomorrow)
waus_perf_BT = performance(waus_pred_BT,"tpr","fpr")
plot(waus_perf_BT, add=TRUE, col = "red")

#Computing AUC for Boosting
waus_AUC_BT = performance(waus_pred_BT, "auc")
value4 = as.numeric(waus_AUC_BT@y.values)
value4

#Random Forest
waus_pred_RF_test = predict(waus_randomforest, waus.test, type="prob")
waus_pred_RF = prediction( waus_pred_RF_test[,2], waus.test$RainTomorrow)
waus_perf_RF = performance(waus_pred_RF,"tpr","fpr")
plot(waus_perf_RF, add=TRUE, col = "darkgreen")
# Add a legend to the plot
legend("bottomright",legend=c("Decision Tree", "Naive Bayes", "Bagging", "Boosting", "Random Forest"), 
       fill=c('orange','blue','violet','red','darkgreen'),cex=0.8, text.font=4)


#Computing AUC for Random Forest
waus_AUC_RF = performance(waus_pred_RF, "auc")
value5 = as.numeric(waus_AUC_RF@y.values)
value5



#Q7
column1 = c(confusion_value1,confusion_value2,confusion_value3,confusion_value4,confusion_value5)
column2 = c(value1,value2,value3,value4,value5)
combined_df = data.frame(column1,column2)
names(combined_df) = c('Confusion Matrix Accuracy', 'AUC')

combined_df


#Q8
#Determine the most important variables for each classification model
#Decision Tree
summary(waus_decisiontree)


#Naiva Bayes
waus_naivebayes


#Bagging
waus_bagging$importance
barplot(waus_bagging$importance[order(waus_bagging$importance, decreasing = TRUE)],
        ylim = c(0, 100), main = "Variables Relative Importance",
        col = "lightblue")


#Boosting
waus_boosting$importance
barplot(waus_boosting$importance[order(waus_boosting$importance, decreasing = TRUE)],
        ylim = c(0, 100), main = "Variables Relative Importance",
        col = "lightblue")

#Random Forest
waus_randomforest$importance
barplot(waus_randomforest$importance[order(waus_randomforest$importance, decreasing = TRUE)],
        ylim = c(0, 100), main = "Variables Relative Importance",
        col = "lightblue")



#remove the unimportant attributes in the training dataset, testing dataset and the dataset
waus.train = subset(waus.train, select = -c(Location,WindDir9am,RainToday))
waus.test = subset(waus.test, select = -c(Location,WindDir9am,RainToday))
waus = subset(waus, select = -c(Location,WindDir9am,RainToday))


#Q9
#Decision Tree
#fit tree model on training data
waus_fit = tree(RainTomorrow~., data = waus.train)
summary(waus_fit)

#plot the decision tree out to visualise
plot(waus_decisiontree)
text(waus_decisiontree, pretty = 0)
#test accuracy
waus_predict = predict(waus_fit, waus.test, type = "class")
tree1=table(predicted = waus_predict, actual = waus.test$RainTomorrow)
tree1


#cross valdiation and pruning
test_fit=cv.tree(waus_fit, FUN=prune.misclass)
test_fit
prune_waus_fit = prune.misclass(waus_fit, best=3)
summary(prune_waus_fit)
plot(prune_waus_fit)
text(prune_waus_fit, pretty=0)
#test accuracy after pruning
waus_prune_predict = predict(prune_waus_fit, waus.test, type = "class")
tree2 = table(predicted = waus_prune_predict, actual = waus.test$RainTomorrow)
tree2

#confusion matrix
prune_tree_confusion = confusionMatrix(waus_prune_predict,waus.test$RainTomorrow)
c1 = prune_tree_confusion$overall['Accuracy']
c1

#Bagging
waus_baggingcv = bagging.cv(RainTomorrow ~ ., v = 5, data = waus.train, mfinal = 10, control = rpart.control(maxdepth = 1))
waus_baggingcv$confusion

c2 = (waus_baggingcv$confusion[1,1]+waus_baggingcv$confusion[2,2])/
  sum(waus_baggingcv$confusion)
c2


#Error before cross validation
waus_pred_bagging$error
#Error after cross validation
waus_baggingcv$error



#Boosting
waus_boostcv <- boosting.cv(RainTomorrow ~ ., v = 5, data = waus.train, mfinal = 10,control = rpart.control(maxdepth = 1))
waus_boostcv$confusion

c3 = (waus_boostcv$confusion[1,1]+waus_boostcv$confusion[2,2])/
  sum(waus_boostcv$confusion)
c3

#Error before cross validation
waus_pred_boosting$error

#Error after cross validation
waus_boostcv$error





#Random Forest
waus_rfcv = rfcv(trainx = waus.train[,-c(19)],	trainy = waus.train[,c(19)],	cv.fold=5,	scale="log",	step=0.5)
waus_rfcv

waus_cvrandomforest = randomForest(RainTomorrow~., data = waus.train, na.action = na.exclude, ntree=500, mtry=18)
waus_cvrandomforest
waus_pred_cvRF = predict(waus_cvrandomforest, waus.test)
RF_confusion_matrix = table(Predicted_Class = waus_pred_cvRF, Actual_Class = waus.test$RainTomorrow)
c4 = (RF_confusion_matrix[1,1]+RF_confusion_matrix[2,2])/
  sum(RF_confusion_matrix)
c4


#Q10
library(neuralnet)
library(car)

WAUS = read.csv("WAUS2020.csv")
L = as.data.frame(c(1:49))
set.seed(29634431) # Student ID is the random seed
L = L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
WAUS = WAUS[(WAUS$Location %in% L),]
WAUS = WAUS[sample(nrow(WAUS), 2000, replace = FALSE),] # sample 2000 rows

#Create a new dataset that only stores the important attributes to create the ANN
final_waus = subset(WAUS, select = c(Humidity3pm, Sunshine, WindGustDir, WindDir3pm,Pressure3pm, Cloud3pm, RainTomorrow))
#remove the NA values
final_waus = final_waus[complete.cases(final_waus),]
final_waus
#Convert the RainTomorrow, WindGustDir and WindDir3pm variable into binary columns as indicator variables
final_waus$RainTomorrow = recode(final_waus$RainTomorrow, " 'No' = '0'; 'Yes' = '1' ")
final_waus$RainTomorrow = as.character(final_waus$RainTomorrow)
final_waus$RainTomorrow = as.numeric(final_waus$RainTomorrow)

final_waus$WindGustDir = factor(final_waus$WindGustDir)
final_waus$WindGustDir = recode(final_waus$WindGustDir, " 'E' = '0'; 'ENE' = '1';'ESE' = '2'; 'N' = '3';
                                 'NE' = '4'; 'NNE' = '5'; 'NNW' = '6'; 'NW' = '7'; 'S' = '8'; 'SE' = '9';
                                 'SSE' = '10'; 'SSW' = '11'; 'SW' = '12'; 'W' = '13'; 'WSW' = '14'; 'WNW' = '15' ")
final_waus$WindGustDir = as.character(final_waus$WindGustDir)
final_waus$WindGustDir = as.numeric(final_waus$WindGustDir)

final_waus$WindDir3pm = factor(final_waus$WindDir3pm)
final_waus$WindDir3pm = recode(final_waus$WindDir3pm, " 'E' = '0'; 'ENE' = '1';'ESE' = '2'; 'N' = '3';
                                 'NE' = '4'; 'NNE' = '5'; 'NNW' = '6'; 'NW' = '7'; 'S' = '8'; 'SE' = '9';
                                 'SSE' = '10'; 'SSW' = '11'; 'SW' = '12'; 'W' = '13'; 'WSW' = '14'; 'WNW' = '15' ")
final_waus$WindDir3pm = as.character(final_waus$WindDir3pm)
final_waus$WindDir3pm = as.numeric(final_waus$WindDir3pm)

#Scale the data to normalise the data.
final_waus = as.data.frame(scale(final_waus[c(1,2,3,4,5,6,7)]))

#Divide the data into 70% training set and 30% test set
set.seed(29634431) #Student ID as random seed 
train.row = sample(1:nrow(final_waus), 0.7*nrow(waus))
final_waus.train = final_waus[train.row,]
final_waus.test = final_waus[-train.row,]

#get the ANN with the 6 variables in the fianl_waus.train dataset to predict RainTomorrow
rain_ann = neuralnet(RainTomorrow ~  Humidity3pm + Sunshine + WindGustDir +WindDir3pm + Pressure3pm + Cloud3pm
                     ,final_waus.train, hidden = 6, linear.output = FALSE)

plot(rain_ann)

rain_ann.pred = compute(rain_ann,final_waus.test[c(1,2,3,4,5,6)])
prob = rain_ann.pred$net.result
#get all the prob that is above 0.5 and classified it as 1 and if it's not 0.
rain_ann.pred2 = ifelse(prob>=0.5,1,0)

#print out the confusion matrix table
conftable = table(observed	=	final_waus.test$RainTomorrow,	predicted	=	rain_ann.pred2)
conftable
#get the accuracy of ANN
accuracy = (conftable[1,1]+conftable[2,2])/sum(conftable)
accuracy



