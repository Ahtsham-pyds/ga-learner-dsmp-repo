# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(path)
data.dropna(subset=['Rating'],inplace=True)
plt.hist(data['Rating'],bins=60)

data=data.query('Rating<=5')
plt.hist(data['Rating'],bins=60)



#Code starts here


#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
total_null=pd.Series(total_null)

percent_null=total_null/len(data)
percent_null=pd.Series(percent_null)*100

missing_data=pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])

data.dropna(inplace=True)
total_null_1=data.isnull().sum()
percent_null_1=total_null_1/len(data)
missing_data_1=pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])

print(missing_data_1)
# code ends here


# --------------

#Code starts here
sns.catplot('Category','Rating',data=data,kind='box',height=10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

#Removing `,` from the column
data['Installs']=data['Installs'].str.replace(',','')

#Removing `+` from the column
data['Installs']=data['Installs'].str.replace('+','')

#Converting the column to `int` datatype
data['Installs'] = data['Installs'].astype(int)

#Creating a label encoder object
le=LabelEncoder()

#Label encoding the column to reduce the effect of a large range of values
data['Installs']=le.fit_transform(data['Installs'])

#Setting figure size
plt.figure(figsize = (10,10))

#Plotting Regression plot between Rating and Installs
sns.regplot(x="Installs", y="Rating", color = 'teal',data=data)

#Setting the title of the plot
plt.title('Rating vs Installs[RegPlot]',size = 20)

#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())

data['Price']=data['Price'].str.replace('$','')
data['Price']=data['Price'].astype('float')

sns.regplot('Price','Rating',data=data)
plt.title('Rating vs Price [RegPlot]')


#Code ends here


# --------------

#Code starts here

data['Genres'].unique()
#data['Genres']=(data['Genres'].str.split(';'))[0]
data['Genres']=data['Genres'].str.split(';',1).str[0]

#Code ends here
gr_mean=data.groupby('Genres',as_index=False).mean()
print(gr_mean.describe())

gr_mean=gr_mean.sort_values(by=['Rating'])

print(gr_mean.iloc[0,-1])


# --------------

#Code starts here

import datetime as dt

data['Last Updated']=pd.to_datetime(data['Last Updated'])
data['Last Updated Days']=(data['Last Updated'].max()-data['Last Updated']).dt.days


sns.regplot('Last Updated Days', 'Rating',data=data)
plt.title('Rating vs Updated [RegPlot]')
#Code ends here



