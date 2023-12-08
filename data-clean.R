library(tidyverse)
library(haven)

df1=read_sav("ERCP-20230914内部模型-1012.sav") %>% 
  mutate(type="in")
df2=read_sav("ERCP-多中心外部验证1012.sav"  ) %>% 
  mutate(type="out")

df=rbind(df1,df2)
a=df %>% select(4,9:12,25:28,31:38) %>% 
  mutate_if(is.numeric,as.factor)
b=df %>% select(-colnames(a))
df=cbind(a,b) %>% select(-year,-No,-ID) %>% 
  select(-Npercent,-NippleOpeningMethod,- ALP,-GGT,-ALB)
# 1. （height, weight）和BMI可以二选一，也可以都选；
# 2. Npercent（中性粒细胞百分比）和Nvalue（中性粒细胞数目）只能二选一；
# 3. ALP、GGT和ALB可以保留也可以删掉；
# 4. NippleOpeningMethod与（EST、EPBD）只能二选一。




## 缺失值查看
library(DataExplorer)
#plot_missing(df)



df1=df %>% filter(type=="in")
df2=df %>% filter(!type=="in")

#######################################
### Main outcome is PEP
#######################################
# ALL data by group
library(compareGroups)
res <- compareGroups(type ~ ., data =df)
createTable(res)
export2csv(createTable(res), file='tableall.csv')
xa=read.csv('tableall.csv',header=T) %>% 
  set_names("Variables","Train","Test","p-value")
DT::datatable(xa)

## Train
# ALL data by group
library(compareGroups)
res <- compareGroups(PEP ~ ., data =df1)
createTable(res)
export2csv(createTable(res), file='table1.csv')
xa=read.csv('table1.csv',header=T) %>% 
  set_names("Variables","No","Yes","p-value")
DT::datatable(xa)

## Test
# ALL data by group
library(compareGroups)
res <- compareGroups(PEP ~ ., data =df2)
createTable(res)
export2csv(createTable(res), file='table2.csv')
xa=read.csv('table2.csv',header=T) %>% 
  set_names("Variables","No","Yes","p-value")
DT::datatable(xa)


# Load necessary libraries
library(caret)
library(xgboost)
library(pROC)
library(ROSE)
# Split the data into training and test sets
dfx=df1 %>% fill(HBP, #NippleOpeningMethod,,ALB  ALP,GGT 
                 DiameterofStone 
                 ) %>% select(-type) %>% na.omit() %>% 
  select(-NippleType) %>% 
  mutate(PEP=as.factor(ifelse(PEP==1,"yes","no"))) %>% 
  select(-DB,-BMI) %>% mutate(id=1:n())
# Oversample the minority class using ROSE
dfx <- ROSE(PEP ~ ., data = dfx, seed = 123)$data
table(dfx$PEP)

set.seed(123)
trainIndex <- createDataPartition(dfx$PEP, p = .8, list = FALSE, times = 1)
train <- dfx[trainIndex, ]
test  <- dfx[-trainIndex, ]


# Define a pre-processing model to handle missing values

# Set up the control using a cross-validated method
fitControl <- trainControl(method = "cv",number = 5,savePredictions = "final",classProbs = TRUE,summaryFunction = twoClassSummary)
# Train the model
model <- train(PEP ~ ., data = train,  method = "xgbTree",trControl = fitControl,metric = "ROC")
# Predict on the test set
predictions <- predict(model, newdata = test, type = "prob")
# Compute AUC
roc_obj <- roc(test$PEP, predictions[,2])
auc(roc_obj)

# Compute 95% CI
ci(roc_obj, method = "bootstrap")

save(model,test,train,file = "xgboost2.Rdata")

plot(varImp(model))
# 
# 
# 
# # Load necessary library
# library(ROSE)
# 
# Train the model
model1 <- train(PEP ~ ., data = train, method = "rf", trControl = fitControl)
# Print model details
predictions <- predict(model1, newdata = test, type = "prob")
# Compute AUC
print(model1)
# Compute AUC
roc_obj <- roc(test$PEP, predictions[,2])
auc(roc_obj)
# Compute 95% CI
ci(roc_obj, method = "bootstrap")

# 
# 

