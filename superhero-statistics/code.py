# --------------
#Code starts here
import matplotlib.pyplot as plt
fig = plt.figure()
ax_1=plt.subplot(311)
ax_1.boxplot(data['Intelligence'])
ax_1.set_title('Intelligence')
ax_2=plt.subplot(312)
ax_2.boxplot(data['Speed'])
ax_2.set_title('Speed')
ax_3=plt.subplot(313)
ax_3.boxplot(data['Power'])
ax_3.set_title('Power')
plt.show()


# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(path)
#path of the data file- path

#Code starts here 

data['Gender'].replace('-','Agender',inplace=True)
gender_count=data['Gender'].value_counts()

gender_count.plot(kind='bar')
plt.show()


# --------------
#Code starts here
#total_high=data['Total'].quantile(.95)

total_high=round(data['Total'].quantile(.99),2)
super_best=data.query("Total>554.05")
super_best_names=list(super_best.iloc[:,1])
super_best_names


# --------------
#Code starts here

alignment=pd.Series(data['Alignment'].value_counts())
alignment
alignment.plot(kind='pie')
plt.title('Character ALignment')
plt.show()


# --------------
#Code starts here
sc_df=data[['Strength','Combat']]
#sc_covariance=(((sc_df['Strength'].mean()-sc_df['Strength'])*(sc_df['Combat']#.mean()-sc_df['Combat'])).sum()/len(sc_df)).round(2)
sc_covariance=(np.cov(sc_df['Strength'],sc_df['Combat'])[0,1]).round(2)

sc_strength=round(sc_df['Strength'].std(),2)
sc_combat=round(sc_df['Combat'].std(),2)
sc_pearson=round(sc_covariance/(sc_strength*sc_combat),2)

ic_df=data[['Intelligence','Combat']]
#ic_covariance=(((ic_df['Intelligence'].mean()-ic_df['Intelligence'])*(ic_df['Combat'].mean()-ic_df['Combat'])).sum()/len(ic_df)).round(2)
ic_covariance=(np.cov(ic_df['Intelligence'],ic_df['Combat'])[0,1]).round(2)
ic_intelligence=round(ic_df['Intelligence'].std(),2)
ic_combat=round(ic_df['Combat'].std(),2)
ic_pearson=round(ic_covariance/(ic_combat*ic_intelligence),2)






