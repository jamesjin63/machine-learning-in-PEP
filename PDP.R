
library(rms)
library(plotRCS)
library(tidyverse)
load("/Users/anderson/Desktop/PhD/Parttime/Wanggang/JAMA/realdfx.Rdata")
df1=readxl::read_excel("/Users/anderson/Desktop/zhongjie/df12237.xlsx",1) %>% 
  #mutate(PEP=(ifelse(PEP=="yes",1,0))) %>% 
  mutate(center="main",.before =1) %>% 
  select(1,15,2:14,16:27)

dfx1=read.csv("/Users/anderson/Desktop/ERCP/工作表 1-ERCP-20230914内部模型-11.26-2247例.csv",header = T)
library(caret)
df2=dfx1 %>% select(PEP,PancreatitisHis,DifficultIntubation,
                  age,DiameterofBileDuct,DiameterofStone,TB,
                  gender, AST,EST, ALT,Nvalue,WBC,gallbladder,  Gallstones) %>% 
  mutate(PEP=as.factor(PEP))

table(df2$PEP)
set.seed(143)
trainIndex <- createDataPartition(df2$PEP, p = .8, list = FALSE, times = 1)
train <- df2
test  <- df2[-trainIndex, ]


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
    filter(class=="1")
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

p4=pdpplot(dfall =train,xa="AST",ylimx=0.1)

p5=pdpplot(dfall =train,xa="ALT",ylimx=0.1)

p6=pdpplot(dfall =train,xa="TB",ylimx=0.1)

#p7=pdpplot(dfall =train,xa="WBC",ylimx=0.75)

library(patchwork)
(p1+p2+p3)/
  (p4+p5+p6)
ggsave("PDP.pdf",width = 12,height = 10,dpi=300)




df=haven::read_sav("/Users/anderson/Desktop/PhD/Parttime/Wanggang/JAMA/ERCP-20230914内部模型-11.26-2247例.sav")







