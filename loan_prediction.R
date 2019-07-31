library(readr)
library(caret)
library(tidyverse)
library(psych)
library(party)
library(randomForest)
library(mice)
library(VIM)

#load dataset
df <- read_csv("~/Programming/Dataset/CSV/loan/train_ctrUa4K.csv")
validation<- read_csv("~/Programming/Dataset/CSV/loan/test_lAUu6dG.csv")

#explore dataset
glimpse(df)
summary(df)

#convert columns into factors
cols = c('Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status')
df[cols] = lapply(df[cols], factor)

#gender loan status
gender_status <- table(df$Gender,df$Loan_Status)
summary(gender_status)
plot(gender_status)

#married 
married_status <- table(df$Married,df$Loan_Status)
summary(married_status)
plot(married_status)

#dependant 
dependants_status <- table(df$Dependents,df$Loan_Status)
summary(dependants_status)
plot(dependants_status)

#education 
education_status <- table(df$Education,df$Loan_Status)
summary(education_status)
plot(education_status)

#selfemployed 
selfemployed_status <- table(df$Self_Employed,df$Loan_Status)
summary(selfemployed_status)
plot(selfemployed_status)

#property_area 
area_status <- table(df$Property_Area,df$Loan_Status)
summary(area_status)
plot(area_status)

#property_area 
credithistory_status <- table(df$Credit_History,df$Loan_Status)
summary(credithistory_status)
plot(credithistory_status)

#pairpanels
pairs.panels(df[7:10],bg=df$Loan_Status,pch=21)
featurePlot(x=df[,7:10],y=df[,13],plot="pairs")


ggplot(df,aes(x=df$Loan_Status,y=df$ApplicantIncome))+
  geom_boxplot()+
  facet_grid(df$CoapplicantIncome)

#impute

loan_plot <- aggr(df,col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(df), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))

imputed_Data <- mice(df, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_Data)

completeData <- complete(imputed_Data,2)


#splitting
new_df=select(completeData,-Loan_ID)
trainIndex = createDataPartition(new_df$Loan_Status,p=0.8,times=1,list=F)
training = new_df[trainIndex,]
testing = new_df[-trainIndex,]


#modelling
tree_train <- ctree(Loan_Status~.,data=training)
plot(tree_train)
predict_tree <- predict(tree_train,testing)
tree <- confusionMatrix(predict_tree,testing$Loan_Status,positive = 'Y')
tree
tree$byClass


rf <- randomForest(Loan_Status~Credit_History+ApplicantIncome+LoanAmount+CoapplicantIncome+Loan_Amount_Term,data=training)
predict_rf <- predict(rf,testing)
new_rf <- confusionMatrix(predict_rf,testing$Loan_Status,positive = 'Y')
new_rf
new_rf$byClass


#variable importance
importance(rf)
varImp(rf)
varImpPlot(rf)

glm_model <- glm(Loan_Status~.,data=training)


  
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

set.seed(7)
fit.lda <- train(Loan_Status~., data=training, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Loan_Status~., data=training, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Loan_Status~., data=training, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Loan_Status~., data=training, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Loan_Status~., data=training, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

dotplot(results)


predict_svm = predict(fit.svm,testing)
confusion_svm = confusionMatrix(predict_svm,testing$Loan_Status,positive = 'Y')
confusion_svm$byClass

predict_rf = predict(fit.rf,testing)
confusion_rf = confusionMatrix(predict_rf,testing$Loan_Status,positive = 'Y')
confusion_rf$byClass

#predict validation
cols = c('Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area')
validation[cols] = lapply(validation[cols], factor)

#impute validation data
imputed_validation_Data <- mice(validation, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_validation_Data)
completevalidationData <- complete(imputed_validation_Data,2)

new_predict = predict(rf,completevalidationData)

new_w <- data.frame(validation$Loan_ID,new_predict)


new_w <- new_w %>% 
  rename(
    Loan_ID = validation.Loan_ID,
    Loan_Status = new_predict
  )

write.csv(new_w,file='answer.csv',row.names = FALSE)
