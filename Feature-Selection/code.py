# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path
dataset=pd.read_csv(path)
# Code starts here
dataset.info()

# read the dataset



# look at the first five columns
dataset.columns

# Check if there's any column which is not useful and remove it like the column id
dataset.drop(['Id'],axis=1,inplace=True)

# check the statistical description
dataset.describe()


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols=list(dataset.columns)

#number of attributes (exclude target)
size=len(cols)-1
cols
x=cols[-1]
y=cols[0:-1]
#x-axis has target attribute to distinguish between classes
for i in range(size):
    sns.violinplot(x=x, y=y[i],data=dataset)
    plt.show()

#y-axis shows values of an attribute


#Plot violin for all attributes



# --------------
import numpy
threshold = 0.5

# no. of features considered after ignoring categorical variables

num_features = 10

# create a subset of dataframe with only 'num_features'

subset_train = dataset.iloc[:, :num_features]
cols = subset_train.columns

#Calculate the pearson co-efficient for all possible combinations

data_corr = subset_train.corr()
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(data_corr,vmax=0.8,square=True);

# Set the threshold and search for pairs which are having correlation level above threshold
corr_var_list = []

for i in range(0, num_features):
    for j in range(i+1, num_features):
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_var_list.append([data_corr.iloc[i,j], i, j])

# Sort the list showing higher ones first 
s_corr_list = sorted(corr_var_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))



# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

X=dataset.drop(['Cover_Type'],axis=1)
y=dataset['Cover_Type']
dataset.columns
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.

X_train_temp=StandardScaler().fit_transform(X_train.iloc[:,0:10])
X_train1=numpy.concatenate((X_train_temp,X_train.iloc[:,10:]),axis=1)

X_test_temp=StandardScaler().fit_transform(X_test.iloc[:,0:10])
X_test1=numpy.concatenate((X_test_temp,X_test.iloc[:,10:]),axis=1)



#Standardized
#Apply transform only for non-categorical data
scaled_features_train_df=pd.DataFrame(data=X_train1,index=X_train.index,columns=[list(X_train.columns)])
scaled_features_train_df.head()


scaled_features_test_df=pd.DataFrame(data=X_test1,index=X_test.index,columns=[list(X_test.columns)])

#Concatenate non-categorical data and categorical



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:

skb=SelectPercentile(score_func=f_classif,percentile=20)

predictors=skb.fit_transform(X_train1,y_train)

print(X_train1.shape)
print(predictors.shape)

scores=skb.scores_  
print(scores)

top_k_index=sorted(range(len(scores)),key=lambda i: scores[i],reverse=True)[:predictors.shape[1]]
print(top_k_index)

top_k_predictors=[scaled_features_train_df.columns[i] for i in top_k_index]


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())

model_fit_all_features = clf1.fit(X_train, y_train)

predictions_all_features = model_fit_all_features.predict(X_test)

score_all_features = accuracy_score(y_test, predictions_all_features)

print(score_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], y_train)

predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])

score_top_features = accuracy_score(y_test, predictions_top_features)

print(score_top_features)


