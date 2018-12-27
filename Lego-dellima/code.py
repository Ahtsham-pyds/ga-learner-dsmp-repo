# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df=pd.read_csv(path)
df.head()

X=df.drop(['list_price'],axis=1)
#X=df.iloc[:,[0,2,3,4,5,6,7,8,9]]
y=df.iloc[:,1]

X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.3,random_state=6)

# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
cols=X_train.columns
fig,axes=plt.subplots(nrows=3,ncols=3)

for i in range(3):
    for j in range(3):
        col=cols[3*i+j]
        axes[i,j].scatter(X_train[col],y_train)
    


# code ends here

#for i in range(len(cols)):
#    if i/3<1:
#        for j in range(3):
#            plt.subplot(1,)



# --------------
# Code starts here

corr=X_train.corr()

X_train.drop(['play_star_rating','val_star_rating'],axis=1,inplace=True)
X_test.drop(['play_star_rating','val_star_rating'],axis=1,inplace=True)

# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here

regressor=LinearRegression()

regressor.fit(X_train,y_train)
#print(regressor.coef_)
#print(regressor.intercept_)
# Code ends here
#y1_test=y_test
#y1_test=y1_test.reset_index(drop=True)
y_pred=pd.Series(regressor.predict(X_test))
#result=pd.concat([y1_test,pred],axis=1,keys=['actual','pred'])
#print(result)
#print(pred)
#df.info()

mse=mean_squared_error(y_test,y_pred)
print(mse)

r2=r2_score(y_test,y_pred)
print(r2)


# --------------
# Code starts here


# calculate the residual
residual = (y_test - y_pred)

# plot the figure for residual
plt.figure(figsize=(15,8))
plt.hist(residual, bins=30)
plt.xlabel("Residual")
plt.ylabel("Frequency")   
plt.title("Error Residual plot")
plt.show()

# Code ends here


