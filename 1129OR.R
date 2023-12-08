library(tidyverse)
library(haven)
setwd("/Users/anderson/Desktop/PhD/Parttime/Wanggang/JAMA")
rm(list = ls())
x3=read.csv("outside11.28.csv") 
df=read.csv("2247-Train-test 11.28.csv",header = T) %>% 
  select(-year,-ID,-No)

## 缺失值查看
library(DataExplorer)

df1=df %>% select(1:14) %>% mutate_if(is.numeric,as.factor)
df2=df %>% select(15:27) 
df=cbind(df1,df2)%>% select(-type) %>% 
  mutate(PEP=as.factor(ifelse(PEP=="yes",1,0))) %>% 
  select(14,1:13,15:26)

plot_missing(df)
library(tableone)
T1=CreateTableOne(data=df,strata = c("PEP"))
T3=print(T1,showAllLevels = TRUE,nonnormal = c("bili","platelet"),
         exact = "sex")
write.csv(T3, file = "Table-new1129.csv")
DT::datatable(T3)


df=cbind(df1,df2)
plot_missing(df)
library(tableone)
T1=CreateTableOne(data=df,strata = c("type"))
T3=print(T1,showAllLevels = TRUE,nonnormal = c("bili","platelet"),
         exact = "sex")
write.csv(T3, file = "Table-new1129_train_test.csv")
DT::datatable(T3)



####################################################################################################
# 单因素
uni_glm_out=function(mydata=mydata,y="Y",xva=c("a","B")){
  
  dfglm= mydata %>% dplyr::select(y,xva)
  
  uni_out=tibble() #储存模型结果
  for(i in 1:length(xva)){
    dfx=dfglm %>% select(1,i+1) %>% dplyr::rename("y"=1) %>% mutate(y=as.factor(y))
    glm1<-glm(y~.,data = dfx,family = binomial)
    glm2<-summary(glm1)
    #计算我们所要的指标
    OR<-round(exp(coef(glm1)),3)
    SE<-round(glm2$coefficients[,2],5)  
    CI2.5<-round(exp(coef(glm1)-1.96*SE),3)
    CI97.5<-round(exp(coef(glm1)+1.96*SE),3)
    #CI<-paste0(CI2.5,'-',CI97.5)
    B<-round(glm2$coefficients[,1],3)
    Z<-round(glm2$coefficients[,3],3)
    P<-round(glm2$coefficients[,4],3)
    
    
    ##-
    or=round(exp(cbind(OR = coef(glm1), confint(glm1))),3)
    CI<-paste0(substr(as.character(or[,1]),1,5),
               "(",substr(as.character(or[,2]),1,5),
               '-',substr(as.character(or[,3]),1,5),")")
    
    
    
    #将计算出来的指标制作为数据框
    uni_glm_model<-data.frame(
      'B'=B,       #1
      'SE'=SE,     #2
      'OR'=OR,     #3
      'CI'=CI,     #4
      'Z' =Z,      #5
      'P'=P)[-1,]  #6
    uni_glm_model$characteristics=rownames(uni_glm_model)     #7
    uni_glm_model=uni_glm_model %>% 
      as_tibble() %>% select(7,1:6)
    uni_out=rbind(uni_glm_model,uni_out)
  }
  return(uni_out)
}

#看下结果是啥样子的
xa=uni_glm_out(mydata = df,y="PEP",xva=colnames(df)[-1])


library(forestploter)
tm <- forest_theme(base_size = 10,  #文本的大小
                   # Confidence interval point shape, line type/color/width
                   ci_pch = 15,   #可信区间点的形状
                   ci_col = "#762a83",    #CI的颜色
                   ci_fill = "blue",     #ci颜色填充
                   ci_alpha = 0.8,        #ci透明度
                   ci_lty = 1,         #CI的线型
                   ci_lwd = 1.5,          #CI的线宽
                   ci_Theight = 0.2, # Set an T end at the end of CI  ci的高度，默认是NULL
                   # Reference line width/type/color   参考线默认的参数，中间的竖的虚线
                   refline_lwd = 1,       #中间的竖的虚线
                   refline_lty = "dashed",
                   refline_col = "grey20",
                   # Vertical line width/type/color  垂直线宽/类型/颜色   可以添加一条额外的垂直线，如果没有就不显示
                   vertline_lwd = 1,              #可以添加一条额外的垂直线，如果没有就不显示
                   vertline_lty = "dashed",
                   vertline_col = "grey20",
                   # Change summary color for filling and borders   更改填充和边框的摘要颜色
                   summary_fill = "yellow",       #汇总部分大菱形的颜色
                   summary_col = "#4575b4",
                   # Footnote font size/face/color  脚注字体大小/字体/颜色
                   footnote_cex = 0.6,
                   footnote_fontface = "italic",
                   footnote_col = "red")

b14=xa
b14$or=exp(b14$B)
b14$lorci=exp(b14$B-1.96*b14$SE)
b14$uorci=exp(b14$B+1.96*b14$SE)
b14$` `<- paste(rep(" ", 24), collapse = " ")

y1=forest(b14[,c(1,11,5,7)],
          est = b14$or,
          lower = b14$lorci,
          upper =b14$uorci,
          sizes = 0.4,
          ci_column = 2,
          ref_line = 1,
          #xlim = c(0, 2),
          theme=tm)

library(eoffice)
#topptx(plot(y1),"physical_health.pptx",width = 8,height = 10)
## save
pdf( "y1.pdf",height = 8,width =7)
plot(y1)
dev.off()


####################################################################################################
# 多因素
mul_glm_out=function(mydata=mydata,y="Y",xva=c("a","B")){
  
  dfglm= mydata %>% dplyr::select(y,xva) %>% 
    dplyr::rename("y"=1) %>% mutate(y=as.factor(y))
  modelA = glm(y~., data = dfglm, family = binomial(logit))
  glm3a=modelA
  
  #3.多因素-backward
  modelC<-step(modelA,direction ="backward")
  glm3b=modelC
  
  #4.多因素-both
  modelD<-step(modelA,direction = "both")
  glm3c=modelD
  
  all=list(glm3a,glm3b,glm3c)
  names(all)=c("Enter","backward","both")
  output=tibble()
  for (i in 1:3) {
    glm3=summary(all[[i]])
    OR<-round(exp(glm3$coefficients[,1]),3)
    SE<-round(glm3$coefficients[,2],5)
    print(i)
    
    CI2.5<-round(exp(glm3$coefficients[,1]-1.96*SE),3)
    CI97.5<-round(exp(glm3$coefficients[,1]+1.96*SE),3)
    #CI<-paste0(CI2.5,'-',CI97.5)
    
    B<-round(glm3$coefficients[,1],3)
    Z<-round(glm3$coefficients[,3],3)
    P<-round(glm3$coefficients[,4],3)
    ## OR
    #or=round(exp(cbind(OR = coef(all[[i]]), confint(all[[i]]))),3)
    # CI<-paste0(or[,2],'-',or[,3])
    
    #data.frame(t(confint(all[[i]])))
    
    if(is.null(dim(confint(all[[i]]))[1])){
      or=tibble(OR=coef(all[[i]]),
                low= as.data.frame(t(confint(all[[i]])))[,1],
                up= as.data.frame(t(confint(all[[i]])))[,2]
      )
    } else{
      ## OR
      or=tibble(OR=coef(all[[i]]),
                low= confint(all[[i]])[,1],
                up= confint(all[[i]])[,2]
      )
      
    }
    or=or %>% mutate(CI=paste0(substr(as.character(exp(OR)),1,5),
                               " (",substr(as.character(exp(low)),1,5),
                               '-',substr(as.character(exp(up)),1,5),")"))
    
    
    #制作数据框#'characteristics'=multiname,
    mlogit<-data.frame( 
      #'characteristics'=c("Intercept",xva),#此步可以先不执行，因为常数项尚未去除，
      'B'=B,
      'SE'=SE,
      'OR'=OR,
      'CI'=or$CI,
      'Z' =Z,
      'P'=P,
      "Method"=names(all)[i])#[-1,] 
    mlogit$characteristics=rownames(mlogit)
    mlogit=mlogit %>% 
      as_tibble() %>% select(8,1:7) %>% 
      mutate(characteristics=ifelse(characteristics=="1","(Intercept)",characteristics))
    
    output=rbind(mlogit,output)
  }
  
  output
  return(output)
}

xa %>% filter(P<0.10) %>% select(characteristics) %>% pull()
xva=c("DifficultIntubation","PancreatitisHis"  ,"DiameterofBileDuct"   ,           
      "age","EST" , "DiameterofStone",    "gender"  )
xb=mul_glm_out(mydata = df,y="PEP",xva=xva)


b14=xb
b14$or=exp(b14$B)
b14$lorci=exp(b14$B-1.96*b14$SE)
b14$uorci=exp(b14$B+1.96*b14$SE)
b14=b14 %>% rename('OR (95%CI)'=5)
b14$` `<- paste(rep(" ", 24), collapse = " ")

y2=forest(b14[,c(1,11,5,7)],
          est = b14$or,
          lower = b14$lorci,
          upper =b14$uorci,
          sizes = 0.4,
          ci_column = 2,
          ref_line = 1,
          #xlim = c(0, 2),
          theme=tm)

y2

library(eoffice)
#topptx(plot(y1),"physical_health.pptx",width = 8,height = 10)
## save
pdf( "y2.pdf",height = 4,width =7)
plot(y2)
dev.off()
write.csv(xa,"Single_OR.csv",row.names = F)
write.csv(xb,"Multi_OR.csv",row.names = F)





