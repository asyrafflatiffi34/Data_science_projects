library(tidyverse)
library(caret)
library(party)
library(randomForest)
library(VIM)
library(mice)
library(psych)
library(MASS)
library(Metrics)
library(dplyr)
library(rpart)

#load Data
df <- read.csv("~/Programming/R/3 - Projects/Project 13 - Bigmart/Data/Train_UWu5bXk.txt")
validation <- read.csv("~/Programming/R/3 - Projects/Project 13 - Bigmart/Data/Test_u94Q5KV.txt")

#data exploration
glimpse(df)
summary(df)

plot(df$Outlet_Size,df$Outlet_Type)

loan_plot <- aggr(df,col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(df), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))

scat <- df%>%
  select(Item_Weight,Item_Visibility,Item_MRP,Item_Outlet_Sales)

pairs.panels(scat)


boxplot(df$Item_Outlet_Sales~df$Outlet_Type)
sales_type <- aov(df$Item_Outlet_Sales~df$Outlet_Type)
summary(sales_type)

boxplot(df$Item_Outlet_Sales~df$Outlet_Location_Type)
sales_loc <- aov(df$Item_Outlet_Sales~df$Outlet_Location_Type)
summary(sales_loc)

boxplot(df$Item_Outlet_Sales~df$Outlet_Size)
sales_siz <- aov(df$Item_Outlet_Sales~df$Outlet_Size)
summary(sales_siz)

boxplot(df$Item_Outlet_Sales~df$Item_Fat_Content)
sales_fat <- aov(df$Item_Outlet_Sales~df$Item_Fat_Content)
summary(sales_fat)

boxplot(df$Item_Outlet_Sales~df$Outlet_Identifier)
sales_iden <- aov(df$Item_Outlet_Sales~df$Outlet_Identifier)
summary(sales_iden)

plot(df$Item_Outlet_Sales~df$Item_Weight)
cor(df$Item_Outlet_Sales,df$Item_Weight)


fat_type <-table(df$Item_Fat_Content,df$Outlet_Type)
plot(fat_type)
summary(fat_type)

fat_location <-table(df$Item_Fat_Content,df$Outlet_Location_Type)
plot(fat_location)
summary(fat_location)

fat_size <-table(df$Item_Fat_Content,df$Outlet_Size)
plot(fat_size)
summary(fat_size)

hist(df$Item_Outlet_Sales)
densityplot(df$Item_Outlet_Sales)

news <- na.omit(df$Item_Weight)

hist(df$Item_Weight)
summary(df$Item_Weight)
densityplot(df$Item_Weight)
mean(news)
median(news)
densityplot(news)

plot(df$Item_Weight,df$Item_Outlet_Sales)
cor(df$Item_Weight,df$Item_Outlet_Sales)


table(df$Outlet_Type,df$Outlet_Size)
plot(df$Outlet_Size,df$Item_Outlet_Sales)
aggregate(df$Item_Outlet_Sales,by=list(Category = df$Outlet_Size),FUN=sum)


#impute

table(df$Outlet_Type,df$Outlet_Size)
plot(df$Outlet_Size,df$Item_Outlet_Sales)
aggregate(df$Item_Outlet_Sales,by=list(Category = df$Outlet_Size),FUN=sum)
df$Outlet_Size <- as.character(df$Outlet_Size)
df$Outlet_Size[df$Outlet_Size == ''] <- 'Small'
df$Outlet_Size <- as.factor(df$Outlet_Size)
summary(df$Outlet_Size)

df$Item_Weight[is.na(df$Item_Weight)] = mean(df$Item_Weight, na.rm=TRUE)

#data transformation
summary(df$Item_Visibility)
df$Item_Visibility[df$Item_Visibility==0]<-NA
df$Item_Visibility[is.na(df$Item_Visibility)] = mean(df$Item_Visibility, na.rm=TRUE)

df$Outlet_Establishment_Year <- as.factor(df$Outlet_Establishment_Year)

df$New_Type <- ifelse(df$Item_Type ==c('Soft Drinks','Hard Drinks'),'Drinks',
                      ifelse(df$Item_Type ==c('Health and Hygiene','Household','Others'),'Non-Consume',
                             'Food'))
                      
df$New_Type <- as.factor(df$New_Type)


table(df$Item_Fat_Content,df$New_Type)
df$Item_Fat_Content <- ifelse(df$Item_Fat_Content ==c('reg','Regular'),'Regular','Low Fat')
df$Item_Fat_Content <- ifelse(df$New_Type =='Non-Consume','Non-Consume',df$Item_Fat_Content)
df$Item_Fat_Content <- as.factor(df$Item_Fat_Content)


#data splitting
new_data = df%>% dplyr::select(-Item_Identifier,-Outlet_Identifier)
trainIndex = createDataPartition(new_data$Item_Outlet_Sales,p=0.8,times=1,list=F)
training = new_data[trainIndex,]
testing = new_data[-trainIndex,]



#modelling

model_lm1 <-lm(Item_Outlet_Sales~.,data=training)
summary(model_lm1)
alias(model_lm1)
step <- stepAIC(model_lm1,direction = 'both')
step$anova


predict_1 <- predict(model_lm1,testing)
mse(predict_1,testing$Item_Outlet_Sales)
plot(predict_1,testing$Item_Outlet_Sales,xlab="Predict",ylab='Actual')
abline(a=0,b=1,col='red')

plot(df$Item_Fat_Content,df$Item_Outlet_Sales)

aov(df$Item_Fat_Content,df$Item_Outlet_Sales)

model_lm2 <-lm(Item_Outlet_Sales~Item_Visibility+Item_MRP+
                 Outlet_Identifier+Outlet_Type
                 ,data=training)
summary(model_lm2)

predict_2 <- predict(model_lm2,testing)
mse(predict_2,testing$Item_Outlet_Sales)
plot(predict_2,testing$Item_Outlet_Sales,xlab="Predict",ylab='Actual')
abline(a=0,b=1,col='red')

model_lm3 <-lm(Item_Outlet_Sales~Outlet_Type+Outlet_Size+Outlet_Location_Type+Item_MRP,data=training)
step3 <- stepAIC(model_lm3,direction = 'both')
step3$anova
summary(model_lm3)

predict_3 <- predict(model_lm3,testing)
mse(testing$Item_Outlet_Sales,predict_3)
plot(predict_3,testing$Item_Outlet_Sales,xlab="Predict",ylab='Actual')
abline(a=0,b=1,col='red')


model_lm4 <- lm(Item_Outlet_Sales~Item_MRP+
                  Outlet_Identifier+Outlet_Type
                ,data=training)

alias(model_lm4)
summary(model_lm4)
anova(model_lm4)
plot(model_lm4)

predict_4 <- predict(model_lm4,testing)
mse(testing$Item_Outlet_Sales,predict_4)
plot(predict_4,testing$Item_Outlet_Sales,xlab="Predict",ylab='Actual')
abline(a=0,b=1,col='red')

#tree

tree_1 <- rpart(Item_Outlet_Sales~.,data=training)
plot(tree_1)
summary(tree_1)

tree_2 <- rpart(Item_Outlet_Sales~Item_MRP+Outlet_Type+Outlet_Establishment_Year+Item_Visibility+Item_Type+Item_Weight,data=training)
plot(tree_2)
summary(tree_2)
tree_22 <- predict(tree_2,testing)
mse(tree_22,testing$Item_Outlet_Sales)
plot(tree_22,testing$Item_Outlet_Sales)

#validation
valid_3 <- predict(model_lm3,validation)
answer <- data.frame(validation$Item_Identifier,validation$Outlet_Identifier,
                   valid_3)

answer <- answer %>% 
  rename(
    Item_Identifier = validation.Item_Identifier,
    Outlet_Identifier = validation.Outlet_Identifier,
    Item_Outlet_Sales = valid_3
  )


write.table(answer, file = "answer.csv", sep = ",",
            row.names = FALSE)


#tree validation
validation$Outlet_Establishment_Year <- as.factor(validation$Outlet_Establishment_Year)

tree_validation <- predict(tree_2,validation)
answer <- data.frame(validation$Item_Identifier,validation$Outlet_Identifier,
                     tree_validation)

answer <- answer %>% 
  rename(
    Item_Identifier = validation.Item_Identifier,
    Outlet_Identifier = validation.Outlet_Identifier,
    Item_Outlet_Sales = tree_validation
  )


write.table(answer, file = "answertree.csv", sep = ",",
            row.names = FALSE)
