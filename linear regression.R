data=read.table("C:\\Users\\Nhu Nguyen\\Dropbox\\document\\wsu\\studying\\Fall 2018\\5030\\finalproject\\data.txt")
data=data.frame(data)
#summary(data)
dim(data)
# There are 11934 observations and There are 18 varibales, that contains 16 features and 2 values we will predict. So I will delete the last one (that means I just predict one value via 16 features)
data=data[,-18]
dim(data)
head(data,n=5)
# I will predict GT Compressor decay state coefficient(V17) and renamed as y via the followings
# Lever position (lp)---(V1) and renamed as x1 
#Ship speed (v) [knots]--- (V2) and renamed as x2
#Gas Turbine (GT) shaft torque (GTT) [kN m]--- (V3) and renamed as x3 
#GT rate of revolutions (GTn) [rpm]--- (V4) and renamed as x4
#Gas Generator rate of revolutions (GGn) [rpm]--- (V5) and renamed as x5
#Starboard Propeller Torque (Ts) [kN] ---(V6) and renamed as x6
#Port Propeller Torque (Tp) [kN] ---(V7) and renamed as x7
#Hight Pressure (HP) Turbine exit temperature (T48) [C] --- (V8) and renamed as x8
#GT Compressor inlet air temperature (T1) [C] --- (V9) and renamed as x9
#GT Compressor outlet air temperature (T2) [C] --- (V10) and renamed as x10
#HP Turbine exit pressure (P48) [bar] --- (V11) and renamed as x11
#GT Compressor inlet air pressure (P1) [bar] --- (V12) and renamed as x12
#GT Compressor outlet air pressure (P2) [bar] --- (V13) and renamed as x13
#GT exhaust gas pressure (Pexh) [bar] --- (V14) and renamed as x14
#Turbine Injecton Control (TIC) [%] --- (V15) and renamed as x15
#Fuel flow (mf) [kg/s] --- (V16) and renamed as x16
colnames(data)=c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","y")



#First, It is easy to see that there is no categorical variable

# Let me omit the na values
data=na.omit(data)
# In fact, the above function does not work if the NA values are in the form "?". However, I have checked by converting the values to numeric to make "?" become "NA". But there is no Problem so I delete these codes.

# To proceed, I will do pre-processing for the data before deviding them into Traning and Validation test to fit the data.
# Let's start
fit1=lm(y~., data=data)
summary(fit1)
# The coefficients in x7,x9,x12 return NA. That means they are linearly dependent on other variables. We will not use it. Set the new data
#data=data[,-c(7,9,12)]
data2=data[,-c(7,9,12)]
fit2=lm(y~., data=data2)
summary(fit2)
plot(fit2)

# Check the influent points
which(cooks.distance(fit2)>1)
# There are no influent points

# check Outlier
library(MASS)
which(studres(fit2)>4)
# there is no outlier

# To proceed, I will check the multi-colinearity
library(car)
vif(fit2)
# Ohhhh, all variables have problem. Let me plot the data to see more clearly
#paris(data)
# The "paris" function may take so much time and in fact, it is not necessary. The data may have linear structure. To delele variables, I will use the stepwise selection methods
library("leaps")
leaps =regsubsets(y~., data = data2)
par(mfrow =c(1, 2))
plot(leaps, scale = "adjr2")
plot(leaps, scale = "bic")
# Let we consider the model with highest adj.r.squared and compare with other models by using cross-validation
# I prefer to use Cross-Validation than anything else (AIC, BIC or Cp,...)

n=length(data$y)
n
train=sample(1:n,round(3*n/4),replace = FALSE)
# It is better to use k-fold validation. However, I will run code again several times before making decision. In addition, the number of observation is large. So, it is still OK.

train2=data2[train,]
test2=data2[-train,]

newfit1=lm(y~x1+x2+x5+x6+x10+x11+x15+x16,data=train2)# Uses 8 variables
mean((test2$y - predict.lm(newfit1,test2)) ^ 2)# MSTE ~ 3e-5


newfit2=lm(y~x4+x8+x10+x11+x16,data=train2)# uses 5 variables
mean((test2$y - predict.lm(newfit2,test2)) ^ 2)# MSTE~5e-5

newfit3=lm(y~x10+x13,data=train2)#uses 2 variables
mean((test2$y - predict.lm(newfit3,test2)) ^ 2)#MSTE ~ 1e-4

newfit4=lm(y~x10,data=train2)# uses 1 variable
mean((test2$y - predict.lm(newfit4,test2)) ^ 2)# MSTE ~ 2e-4

summary(data$y)
mean((test2$y - 0.975) ^ 2)
# There is a fact here, the newfit4 is just as good as a constant predictor (y=mean(data$y)).


# Reference
# Depending on different tasks, we will use different fit models. 
# If We need almost exact predicts, the newfit2 is a good choise. (I recommend newfit2 instead of newfit1). 
# If the error of pridict values can be sightly relaxed, the good reference is newfit3


# Let see some their informations (newfit2 and newfit3)
summary(newfit2)
plot(newfit2)
crPlots(newfit2)
# the fitted value and residuals plot suggest the interactions. 
newfit2_=lm(y~x4*x8*x10*x11*x16,data=train2)
summary(newfit2_)
plot(newfit2_)
# The plots are very good. Let me check the MSTE
mean((test2$y - predict.lm(newfit2_,test2)) ^ 2)
# Wow, It is verygood.

# what about newfit3
summary(newfit3)
plot(newfit3)
crPlots(newfit3)
# let us try to add the interaction
newfit3_=lm(y~x10*x13,data=train2)
summary(newfit3_)
plot(newfit3_)
mean((test2$y - predict.lm(newfit3_,test2)) ^ 2)

# Conclusion
#newfit2_ uses 5 variables, MSTE ~ 3e-6
#newfit3_ uses 2 variables, MSTE~ 8e-5

# My opinion

#best model is newfit2_, uses 5 variables, MSTE~3e-6

# Final product
bestfit=lm(y~x4*x8*x10*x11*x16,data=data)
summary(bestfit)
plot(bestfit)
# Delete the outlier and influent points again. Note that in the first step, we delete the influent points and outlier for the MLR
which(cooks.distance(bestfit)>1)
# there is no influent point
index=which(studres(bestfit)>4)
datanew=data[-index,]
# Construct again the best model.
bestfit=lm(y~x4*x8*x10*x11*x16,data=datanew)
summary(bestfit)
plot(bestfit)
# this is final product