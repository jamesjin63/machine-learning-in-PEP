library(tidyverse)
library(haven)
rm(list = ls())
x3=read.csv("outside11.28.csv") 
df=read.csv("2247-Train-test 11.28.csv",header = T) %>% 
  select(-year,-ID,-No)

## 缺失值查看
library(DataExplorer)
plot_missing(x3)
df1=df %>% select(1:14) %>% mutate_if(is.numeric,as.factor)
df2=df %>% select(15:27) 
df=cbind(df1,df2)

#######################################
### Main outcome is PEP
#######################################
# ALL data by group
library(compareGroups)
res <- compareGroups(PEP ~ ., data =df)
createTable(res)
export2csv(createTable(res), file='table2247.csv')
xa=read.csv('table2247.csv',header=T) %>% 
  set_names("Variables","Train","Test","p-value")
DT::datatable(xa)

## Train
# ALL data by group
library(compareGroups)
res <- compareGroups(type ~ ., data =df)
createTable(res)
export2csv(createTable(res), file='table-train-test.csv')
xa=read.csv('table-train-test.csv',header=T) %>% 
  set_names("Variables","No","Yes","p-value")
DT::datatable(xa)

## Test
# ALL data by group
# Load necessary libraries
library(caret)
library(xgboost)
library(pROC)
library(ROSE)
table(df$PEP)
# Split the data into training and test sets
dfx=df %>% select(-type)
  #mutate(PEP=as.factor(ifelse(PEP==1,"yes","no"))) 
# Oversample the minority class using ROSE
dfROSE <- ROSE(PEP ~ ., data = dfx, seed = 123)$data
table(dfROSE$PEP)

###################################
##### Lasso selection
library(glmnet)
dfx=df %>% select(-type) %>% 
  mutate(PEP=as.factor(ifelse(PEP=="yes",1,0))) %>% 
  select(14,1:13,15:26)

#write.csv(x,file='xdata.csv')
# Convert factors to dummy variables, except for the response variable
dfx_matrix <- model.matrix(~ . -1 -PEP, data = dfx)
# The response variable 'PEP' should be a binary numeric vector
# Converting factor to numeric: levels "0" to 0 and "1" to 1
# Assuming that the first level is "0" and the second level is "1"
# Fit the Lasso model
set.seed(123) # For reproducibility of random results
cv_fit <- cv.glmnet(dfx_matrix, PEP_numeric, alpha = 1, family = 'binomial')
# Run cross-validation & select lambda
#————————————————
# Fit the Lasso model
set.seed(123) # For reproducibility of random results
cv.fit <- cv.glmnet(dfx_matrix, PEP_numeric, alpha = 1, family = 'binomial')
# The best lambda value
best_lambda <- cv.fit$lambda.min

#如果取最小值时,要输出的变量名称
cv.fit$lambda.min
Coefficients <- coef(fit, s = cv.fit$lambda.min)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]
Active.Index
ac1=Active.Coefficients
out1=row.names(Coefficients)[Active.Index]
out1


###################################
##### Save for modeling in Python 
df1=dfROSE  %>%   select(14,1:13,15:26) %>% 
  mutate(PEP=as.factor(ifelse(PEP=="yes",1,0))) %>% 
  #mutate(PEP=(ifelse(PEP=="yes",1,0))) %>% 
  mutate(center="main",.before =1) 

df3=x3 %>% select(colnames(df1))
dfall=rbind(df1,df3) %>% 
  fill(height ,weight ,HB, PLT,TB,AST,DiameterofBileDuct,DiameterofStone  )
dfall %>% group_by(center,PEP) %>% 
  summarise(n = n()) %>% as.data.frame()
write.csv(dfall ,"/Users/anderson/Desktop/df2921(unbalance).csv",row.names = F)




####################################################################################################
library(caret)
set.seed(143)
train <- dfx



#构建模型
# Random Forest with cross-validation
set.seed(123)
ctrl <- trainControl(method = "cv", 
                     number = 5, 
                     repeats = 10, 
                     verboseIter = FALSE)
model_rf <- train(PEP ~ ., 
                  data = train, method = "rf", 
                  trControl = ctrl)
plot(varImp(model_rf))


#训练集预测概率
library(iml)
mod <- Predictor$new(model_rf, data = train, type ="prob")
####################################################################################################
library(pdp)
# Compute the accumulated local effects for the first feature
pdpplot=function(dfall = df_imputed,xa="xdf",ylimx=0.5){
  
  eff <- FeatureEffect$new(mod, feature = xa, method = "pdp")
  x=eff$plot()
  ya=x$data %>% 
    set_names("ALB","class","value","type") %>% 
    filter(class=="yes")
  ya2=x$plot_env$rug.dat %>% select(xa,.type,.value) %>% 
    set_names("ALB","type","value")
  p1=ggplot(ya,aes(x = ALB, y = value))+
    geom_line()+ylim(0,ylimx)+#xlim(0,300)+
    geom_rug( data=ya2,aes(x = ALB, y = value),alpha = 0.2,
              sides="b",position = "jitter") +
    labs(x=xa,y="Probability of PEP")+
    theme_bw()
  write.csv(ya,paste0("PDP",xa,".csv"))
  return(p1)
}

p1=pdpplot(dfall =train,xa="age",ylimx=0.1)

p2=pdpplot(dfall =train,xa="DiameterofBileDuct",ylimx=0.1)

p3=pdpplot(dfall =train,xa="DiameterofStone",ylimx=0.1)
# 
p4=pdpplot(dfall =train,xa="AST",ylimx=0.1)
# 
 p5=pdpplot(dfall =train,xa="TB",ylimx=0.1)
# 
p6=pdpplot(dfall =train,xa="Nvalue",ylimx=0.1)
# 
p7=pdpplot(dfall =train,xa="WBC",ylimx=0.1)

library(patchwork)
(p1+p2+p3)#/(p4+p5+p6+p7)
ggsave("PDP.pdf",width = 10,height = 6,dpi=300)

(p4+p5)/(p6+p7)
ggsave("PDP-sup.pdf",width = 10,height = 8,dpi=300)

####################################################################################################
# 单因素




####################################################################################################
# 多因素
































## roc
setwd("/Users/anderson/Desktop/PhD/Parttime/Wanggang/JAMA/py")
roc1=read.csv('roc_data_Decision Tree.csv', header = T) %>% 
  mutate(type="DT:AUC=0.708 95% CI(0.668-0.751)")
roc2=read.csv('roc_data_GLM.csv', header = T) %>% 
  mutate(type="GLM:AUC=0.744 95% CI(0.693-0.789)")
roc3=read.csv('roc_data_Naive Bayes.csv', header = T) %>% 
  mutate(type="NB:AUC=0.769 95% CI(0.726-0.808)")
roc4=read.csv('roc_data_Random Forest.csv', header = T) %>% 
  mutate(type="RF:AUC=0.947 95% CI(0.927-0.963)")
roc5=read.csv('roc_data_SVM.csv', header = T) %>% 
  mutate(type="SVM:AUC=0.695 95% CI(0.648-0.740)")
roc6=read.csv('roc_data_XGBoost.csv', header = T) %>% 
  mutate(type="XGBOOST:AUC=0.905 95% CI(0.877-0.930)")

roc=rbind(roc1,roc2,roc3,roc4,roc5,roc6)

ggplot(roc)+
  geom_line(aes(FPR,TPR,color=type))+
  geom_abline(lty = 2, alpha = 0.5,color = "gray50",size = 1.0) +
  labs(x="1 - Specificity",y="Sensitivity")+
  theme_bw()+theme(legend.position = c(0.75,0.3))+
  theme(legend.title=element_blank())+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
setwd("/Users/anderson/Desktop/PhD/Parttime/Wanggang/JAMA")

ggsave("roc.pdf",width =8,height = 6,dpi = 300)




## heatmap
## roc
setwd("/Users/anderson/Desktop/PhD/Parttime/Wanggang/JAMA/py")
roc1=read.csv('results-DT.csv', header = T) %>% 
  mutate(type="DT")
roc2=read.csv('results-lm.csv', header = T) %>% 
  mutate(type="GLM")
roc3=read.csv('results-nb.csv', header = T) %>% 
  mutate(type="NB")
roc4=read.csv('results-rf.csv', header = T) %>% 
  mutate(type="RF")
roc5=read.csv('results-svm.csv', header = T) %>% 
  mutate(type="SVM")
roc6=read.csv('results-xgboost.csv', header = T) %>% 
  mutate(type="XGBOOST")

roc=rbind(roc1,roc2,roc3,roc4,roc5,roc6) %>% 
  filter(!Center =="main")

# ======================================================== 
# Plot 
# ======================================================== 
ggplot(data = roc, aes(x = Center, y = type)) + 
  geom_tile(aes(fill = AUC), color = "white", size = 0.2) + 
  scale_fill_gradient(low = "gray95", high = "red") + 
  #scale_fill_distiller(palette = "Spectral")+
  labs(x="Centers",y="Models")+
  theme_grey(base_size = 10) + 
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        panel.grid.major = element_line(colour = "gray90", size = 0.1), # Change colour and size as needed
        panel.grid.minor = element_line(colour = "gray90", size = 0.1), # Change colour and size as needed
        plot.title = element_text(size = 12, colour = "gray50"))

setwd("/Users/anderson/Desktop/PhD/Parttime/Wanggang/JAMA/py")
ggsave("AUC-heat.pdf",width =8,height = 6,dpi = 300)
write.csv(roc,"AUC-heatmap.csv")










