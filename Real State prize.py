import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import csv
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
raw_data=pd.read_csv('C:\\Users\\Vinamra\\Desktop\\Machine Learning\\Linear Regression\\Dataset\\Car price.csv')
print(raw_data.head())
print(raw_data.describe())

#include='all' use to showw all the data because only describe() shows only numerical data

print(raw_data.describe(include='all'))

data=raw_data.drop(['Model'],axis=1)
print(data.describe(include='all'))

#checking the missing values
#true mean 1 and flase means zero
#true=null
#false=not null
print(data.isnull())

#sum fo missing values

print(data.isnull().sum())

#if the missing valuses is<5% of total then you are free to remove those miising values

#drop the row ehich contains Na
data_no_nv=data.dropna(axis=0)
print(data_no_nv.describe(include='all'))

#plot the probility distribution 

sns.distplot(data_no_nv['Price'])






#We saw that some price is to high so we create the line closer to price
#so we have to remove outliers
q=data_no_nv['Price'].quantile(0.99)
#0.99 means 99 percentile

data_1= data_no_nv[data_no_nv['Price']<q]

print(data_1.describe(include='all'))


sns.distplot(data_1['Price'])
# we can see that outliers has been removed thats why we use quantile method

sns.distplot(data_no_nv['Mileage'])

#Same issue with MILAGE an Eng. VOLUME
#So we perfirm similar procedure for both things


q=data_1['Mileage'].quantile(0.99)
data_2= data_1[data_1['Mileage']<q]

#improve result
sns.distplot(data_2['Mileage'])


data_3= data_2[data_2['EngineV']<6.5]

sns.distplot(data_3['EngineV'])

#similarly for year

q=data_3['Year'].quantile(0.01)
data_4= data_3[data_3['Year']>q]   
sns.distplot(data_4['Year'])


#now our data is ready so we reset the data and we will include drop equals to true
#to completely forgot the old index

data_clean=data_4.reset_index(drop=True)

print(data_clean.describe(include='all'))
        
#Checking the OLS assumption
#plot the scatter plot
f,(ax1,ax2,ax3)=plt.subplots(1,3, sharey=True, figsize=(15,3))
ax1.scatter(data_clean['Year'],data_clean['Price'])
ax1.set_title('Price & Year') 
ax2.scatter(data_clean['EngineV'],data_clean['Price'])
ax2.set_title('Price & EngineV')
ax3.scatter(data_clean['Mileage'],data_clean['Price'])
ax3.set_title('Price & Mileage')
plt.show()  

#We can see that the relation of price with feature is exponetial not linear
#so we have to traansform into linear by using numpy.log()

log_Price=np.log(data_clean['Price'])
data_clean['log_Price']=log_Price



f,(ax1,ax2,ax3)=plt.subplots(1,3, sharey=True, figsize=(15,3))
ax1.scatter(data_clean['Year'],data_clean['log_Price'])
ax1.set_title('Log Price & Year') 
ax2.scatter(data_clean['EngineV'],data_clean['log_Price'])
ax2.set_title('Log Price & EngineV')
ax3.scatter(data_clean['Mileage'],data_clean['log_Price'])
ax3.set_title('Log Price & Mileage')
plt.show()

#now we got the linear pattern 
#then we have to remove original variable 
#from the data frame since it is no longer needed

data_cleaned=data_clean.drop(['Price'],axis=1)


#Multicollinearity

print(data_cleaned.columns.values)

variables=data_cleaned[['Mileage','EngineV','Year']]
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif["features"]=variables.columns

print(vif)

data_no_Multicollinearity=data_cleaned.drop(['Year'],axis=1)


#Create Dummy variables

Data_with_Dummies=pd.get_dummies(data_no_Multicollinearity,drop_first=True)

print(Data_with_Dummies.head())    


#Rearrange the data

print(Data_with_Dummies.columns.values)

cols=['log_Price','Mileage' ,'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
 'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota' ,'Brand_Volkswagen',
 'Body_hatch' ,'Body_other' ,'Body_sedan', 'Body_vagon', 'Body_van',
 'Engine Type_Gas' ,'Engine Type_Other' ,'Engine Type_Petrol',
 'Registration_yes']

data_perprocessed=Data_with_Dummies[cols]
print(data_perprocessed.head())


#Create the Regression Model


#in order to create the Regression we need to declare Trget & inputs

target=data_perprocessed['log_Price']
inputs=data_perprocessed.drop(['log_Price'],axis=1)

#Scale the data

scaler=StandardScaler()
scaler.fit(inputs)

#transform the data

input_scaled=scaler.transform(inputs)

#spilts the data in train & test


x_train,x_test,y_train,y_test=train_test_split(input_scaled,target,test_size=0.2,random_state=365)

#test_size=0.2 means we split our data into 80 20

reg=LinearRegression()
reg.fit(x_train,y_train)

y_predicted=reg.predict(x_train)

plt.scatter(y_train,y_predicted)
plt.xlabel("target",size=18)
plt.ylabel("Prediction",size=18)
plt.xlim(6,13) 
plt.ylim(6,13)
plt.show()       


  


















