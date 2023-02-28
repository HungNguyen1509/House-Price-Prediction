library(MASS)
library(dplyr)
library(ggplot2)
library(ISLR)
library(plyr)
library(stringr)
library(tidyr)
library(readr)
library(glmnet)
library(xtable)
library(plsr)
library(splines)
library(pls)
library(tree)
library(randomForest)
data <- read_csv("D:/COURSE/Math 448/perth.csv.csv")
View(data) 
dim(data) #counting how many rows and columns in this data
summary(data) #data summary
hist(data$PRICE) #how frequently for the house's price, the graph's skewness
boxplot(data$PRICE)
data1 <-data[-1:-2]
data2 <-data1[-8:-9]
dataclean <-data2[-11:-13] #filter variables and keep the variables that is necessary
dataclean$PRICE <- as.numeric((dataclean$PRICE))
dataclean$BEDROOMS <- as.numeric((dataclean$BEDROOMS))
dataclean$BATHROOMS <- as.numeric((dataclean$BATHROOMS))
dataclean$GARAGE <- as.numeric((dataclean$GARAGE))
dataclean$LAND_AREA <- as.numeric((dataclean$LAND_AREA))
dataclean$FLOOR_AREA <- as.numeric((dataclean$FLOOR_AREA))
dataclean$BUILD_YEAR <- as.numeric((dataclean$BUILD_YEAR))
dataclean$NEAREST_STN_DIST <- as.numeric((dataclean$NEAREST_STN_DIST))
dataclean$POSTCODE <-as.character(dataclean$POSTCODE)
dataclean$NEAREST_SCH_DIST <- as.numeric(dataclean$NEAREST_SCH_DIST)
dataclean$NEAREST_SCH_RANK <- as.numeric(dataclean$NEAREST_SCH_RANK)
dataclean$DATE_SOLD <- as.character(dataclean$DATE_SOLD)
sum(is.na(dataclean)) #check the total of missing value
summary(dataclean) #look at the missing value in each variable
dataclean <-dataclean %>%
  na.omit() #exclude the missing value
sum(is.na(dataclean))
dim(dataclean)
summary(dataclean)
fin<-subset(dataclean,LAND_AREA <= 500000 & FLOOR_AREA >=50) #keep all the 
observations from dataclean which "LAND_AREA" are less than 500,000 and keep all the 
observations from dataclean1 which "FLOOR_AREA" are more than 50
dim(fin)
summary(fin)
View(fin)
hist(fin$PRICE, main = "HOUSE PRICE", xlab = "PRICE", col = "red")
class(fin$POSTCODE)
### setting a training and test set (90 - 10)
set.seed(159)
train = sample(0.9*nrow(fin))
fin.train <- fin[train,] #training set
fin.test <- fin[-train,] #test set
################################
# Multiple Linear Regression ###
################################
lm.fit = lm(PRICE~BEDROOMS + BATHROOMS + GARAGE + LAND_AREA + 
              FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + NEAREST_SCH_RANK + 
              NEAREST_SCH_DIST,data = fin.train)
summary(lm.fit)
coef(lm.fit)
confint(lm.fit)
lm.pred <- predict(lm.fit,fin.test)
lm.pred
MSE.lm <- mean((lm.pred-fin.test$PRICE)^2) #Test MSE
MSE.lm
sqrt(MSE.lm)
summary(lm.fit)$sigma
summary(lm.fit)$r.sq
pairs(PRICE~BEDROOMS + BATHROOMS + GARAGE + LAND_AREA + 
        FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + NEAREST_SCH_RANK + 
        NEAREST_SCH_DIST,data = fin.train)
AIC(lm.fit)
par(mfrow=c(2,2))
plot(lm.fit)
################################
### Ridge regression ###
################################
set.seed(159)
train.mat <- model.matrix(PRICE ~ BEDROOMS + BATHROOMS + GARAGE + 
                            LAND_AREA + FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + 
                            NEAREST_SCH_RANK + NEAREST_SCH_DIST, data = fin.train)
test.mat <- model.matrix(PRICE ~ BEDROOMS + BATHROOMS + GARAGE + 
                           LAND_AREA + FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + 
                           NEAREST_SCH_RANK + NEAREST_SCH_DIST, data = fin.test)
grid <- 10 ^ seq(4, -2, length = 100)
ridge.fit <- glmnet(train.mat, fin.train$PRICE, alpha = 0, lambda = grid, thresh = 1e-12)
ridge.cv <- cv.glmnet(train.mat, fin.train$PRICE, alpha = 0, lambda = grid, thresh = 1e-12)
bestlam.ridge <- ridge.cv$lambda.min
bestlam.ridge
ridge.pred <- predict(ridge.fit, s = bestlam.ridge, newx = test.mat)
MSE.ridge <- mean((ridge.pred - fin.test$PRICE)^2)
MSE.ridge
sqrt(MSE.ridge)
summary(ridge.fit)
plot(ridge.cv)
################################
### The LASSO ###
################################
set.seed(159)
train.mat <- model.matrix(PRICE ~ BEDROOMS + BATHROOMS + GARAGE + 
                            LAND_AREA + FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + 
                            NEAREST_SCH_RANK + NEAREST_SCH_DIST, data = fin.train)
test.mat <- model.matrix(PRICE ~ BEDROOMS + BATHROOMS + GARAGE + 
                           LAND_AREA + FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + 
                           NEAREST_SCH_RANK + NEAREST_SCH_DIST, data = fin.test)
lasso.fit <- glmnet(train.mat, fin.train$PRICE, alpha = 1, lambda = grid, thresh = 1e-12)
lasso.cv <- cv.glmnet(train.mat, fin.train$PRICE, alpha = 1, lambda = grid, thresh = 1e-12)
bestlam.lasso <- lasso.cv$lambda.min
bestlam.lasso
lasso.pred <- predict(lasso.fit, s = bestlam.lasso, newx = test.mat)
MSE.lasso <- mean((lasso.pred - fin.test$PRICE)^2)
sqrt(MSE.lasso)
plot(lasso.cv)
################################
### PLS ###
################################
par(mfrow=c(1,2))
set.seed(159)
fit.pls <- plsr(PRICE ~ BEDROOMS + BATHROOMS + GARAGE + LAND_AREA + 
                  FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + NEAREST_SCH_RANK + 
                  NEAREST_SCH_DIST, data = fin.train, scale = TRUE, validation = "CV")
validationplot(fit.pls, val.type = "MSEP")
summary(fit.pls)
pred.pls <- predict(fit.pls, fin.test)
MSE.pls<-mean((pred.pls - fin.test$PRICE)^2)
sqrt(MSE.pls)
################################
### PCR ###
################################
set.seed(159)
fit.pcr <- pcr(PRICE ~ BEDROOMS + BATHROOMS + GARAGE + LAND_AREA + 
                 FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + NEAREST_SCH_RANK + 
                 NEAREST_SCH_DIST, data = fin.train, scale = TRUE, validation = "CV")
validationplot(fit.pcr, val.type = "MSEP")
summary(fit.pcr)
pred.pcr <- predict(fit.pcr, fin.test)
MSE.pcr <- mean((pred.pcr - fin.test$PRICE)^2)
sqrt(MSE.pcr)
#comparision pls and pcr
ggplot(fit.pls)
################################
### decision tree ###
################################
tree <- tree(PRICE ~ BEDROOMS + BATHROOMS + GARAGE + LAND_AREA + 
               FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + NEAREST_SCH_RANK + 
               NEAREST_SCH_DIST + POSTCODE, data = fin.train)
summary(tree)
plot(tree)
text(tree, pretty = 0)
pred.tree <- predict(tree, fin.test)
MSE.tree <- mean((fin.test$PRICE - pred.tree)^2)
sqrt(MSE.tree)
#pruning the tree
cv.tree <- cv.tree(tree, FUN=prune.tree)
par(mfrow=c(1, 1))
plot(cv.tree$size, cv.tree$dev, type="b")
#with 9 nodes give the lowest CV error means that this model cannot be prunned.
#check the CV error with 6 nodes
prune.tree <- prune.tree(tree, best = 6)
par(mfrow = c(1, 1))
plot(prune.tree)
text(prune.tree, pretty = 0)
pred.prunetree <- predict(prune.tree, fin.test)
MSE.prunetree <- mean((fin.test$PRICE - pred.prunetree)^2)
sqrt(MSE.prunetree)
################################
## Bagging ##
################################
set.seed(159)
bag <- randomForest(PRICE ~ BEDROOMS + BATHROOMS + GARAGE + 
                      LAND_AREA + FLOOR_AREA + BUILD_YEAR + NEAREST_STN_DIST + 
                      NEAREST_SCH_RANK + NEAREST_SCH_DIST + POSTCODE, data=fin.train, 
                    importance=TRUE)
bag
pred.bag = predict(bag, fin.test)
MSE.bag <- mean((fin.test$PRICE - pred.bag)^2)
sqrt(MSE.bag)
importance(bag)
#comparision
test.avg <- mean(fin.test$PRICE)
lm.r2 <- 1 - mean((lm.pred - fin.test$PRICE)^2) / mean((test.avg - fin.test$PRICE)^2)
ridge.r2 <- 1 - mean((ridge.pred - fin.test$PRICE)^2) / mean((test.avg - fin.test$PRICE)^2)
lasso.r2 <- 1 - mean((lasso.pred - fin.test$PRICE)^2) / mean((test.avg - fin.test$PRICE)^2)
pls.r2 <- 1 - mean((pred.pls - fin.test$PRICE)^2) / mean((test.avg - fin.test$PRICE)^2)
pcr.r2 <- 1 - mean((pred.pcr - fin.test$PRICE)^2) / mean((test.avg - fin.test$PRICE)^2)
tree.r2 <- 1 - mean((pred.tree - fin.test$PRICE)^2) / mean((test.avg - fin.test$PRICE)^2)
bag.r2 <- 1 - mean((pred.bag - fin.test$PRICE)^2) / mean((test.avg - fin.test$PRICE)^2)
print(lm.r2)
print(ridge.r2)
print(lasso.r2)
print(pls.r2)
print(pcr.r2)
print(tree.r2)
print(bag.r2)
all = c(lm.r2, ridge.r2, lasso.r2, pls.r2, pcr.r2, tree.r2, bag.r2)
names(all) = c("lm", "rid", "las", "pls", "pcr", "tree", "bag")
par(mfrow = c(1,2))
barplot(all,main = "test R-Squared",col = "green")
all2 = c(sqrt(MSE.lm), sqrt(MSE.ridge), sqrt(MSE.lasso), sqrt(MSE.pls), sqrt(MSE.pcr), 
         sqrt(MSE.tree), sqrt(MSE.bag))
names(all2) = c("lm", "rid", "las", "pls", "pcr", "tree", "bag")
barplot(all2,main = "test MSE",col = "red")