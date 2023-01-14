# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:18:07 2022

@author: paul
"""
#Let's load all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split, cross_val_score


#Load the train & test data
test = pd.read_csv("house_test.csv")
train = pd.read_csv("house_train.csv")
#Concatenate the test and train sample to avoid the reapeat of data modfication
df = pd.concat([train , test])
#Get our independent variable for prediction
saleprice = df.SalePrice

#Observe the missing value of data
df.isna().sum()[df.isna().sum() > 0]
Total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / df.isnull().count()*100).sort_values(ascending=False)
#Acuire the percentage of NA in each column and see the variables with the most percantage of NA
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])

#Build a correlation matrix an extract the most correlated features with the target variable
corr = df.corr()
highest_corr_features = corr.index[abs(corr["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(df[highest_corr_features].corr(),annot=True,cmap="RdYlGn")
corr["SalePrice"].sort_values(ascending=False)


#Plot their joint distribution
sns.pairplot(df[list(highest_corr_features)].reset_index(drop = True))


#Observe only the targer variable. There is visible heteroskedakcity and skeweness that we have to deal with
pair = sns.PairGrid(df, y_vars= "SalePrice", x_vars=list(highest_corr_features), height=4)
pair.map(sns.regplot)


#Check if any of the most correlated features has lots of NA
for i in list(highest_corr_features):
    if i in list(missing_data[missing_data['Total'] > 5].reset_index()['index']):
        print(i)

#None of the most correlated features has more than 5% of NA, so we can delete columns with more than 5% of NA
df.drop((missing_data[missing_data['Total'] > 5]).index, axis=1, inplace=True)
#Observe the remaining NA in our DataFrame
df.isna().sum()[df.isna().sum() > 0]

#DF only with the columns that have NA
df_with_na = df[list(df.isna().sum()[df.isna().sum() > 0].reset_index()['index'])]

#Divide numerical and categorical features
categor = df.loc[:,df.dtypes==np.object]
numer =df.loc[:,df.dtypes!=np.object]

#For numerical features fill NA with the help of KNN-imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=20)
numer_imp = imputer.fit_transform(numer)
numer_df = pd.DataFrame(numer_imp)
numer_df.columns = numer.columns
#Add a new feature
numer_df['TotalSF'] = numer_df['TotalBsmtSF'] + numer_df['1stFlrSF'] + numer_df['2ndFlrSF']


#For categorical features just fill NA with the mode
categor = categor.apply(lambda x: x.fillna(x.mode()[0]))
#Check for any missing values
categor.isna().sum()[categor.isna().sum() > 0]




#Concatenate modified data frames that contain numeric and categorical features
cols = list(numer_df.columns) + list(categor.columns)
numer_df.reset_index(inplace=True, drop=True)
categor.reset_index(inplace=True, drop=True)
concl = pd.concat([numer_df, categor], axis = 1)
concl.columns = cols




#Also we can witness that or numerical features are not normally distributed
numer_df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

from scipy.stats import norm, skew
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
#Normalise data with logarithm transformation
skewed_feats = numer.apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
skewed_features = high_skew.index

for feat in skewed_features:
    numer[feat] = boxcox1p(numer[feat], boxcox_normmax(numer[feat] + 1))
    
#Apply a one hot encoding method for categorical feature
concl_dum = pd.get_dummies(concl)

#Inspect distribution of our target variable. Does not seem like normal distribution
saleprice.hist()
#Apply a log transformation on it
sale_log = np.log1p(saleprice)


#Divide all the data into train and test sets
train_df = concl_dum.iloc[:len(train),:]
test_df = concl_dum.iloc[len(train):,:]

#Extract the dependent and independetn variables
X = train_df.drop("Id", axis = 1)
y = sale_log.iloc[:len(train)]

#Split the data into train/test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state= 0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
#Apply the RF Regressor 
clf_rf = RandomForestRegressor(random_state=0)
parametrs = {'n_estimators': [i for i in range(10,51, 10)], 'max_depth': [i for i in range(1,12, 2)], "min_samples_leaf" : [i for i in range(1,8)], 'min_samples_split':[i for i in range(2,10, 2)]}
grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=3, n_jobs = -1)
grid_search_cv_clf.fit(X, y)
grid_search_cv_clf.best_params_
feature_importance = grid_search_cv_clf.best_estimator_.feature_importances_
df = pd.DataFrame({'features': list(X_train), 'feature_importance' : feature_importance})
df.sort_values('feature_importance',ascending = False).head(10)


print (grid_search_cv_clf.score(X, y))


ID = test.Id
predicted_prices = grid_search_cv_clf.predict(test_df.drop("Id", axis = 1))
predicted_prices = np.expm1(predicted_prices)

my_submission = pd.DataFrame({'Id': ID, 'SalePrice': predicted_prices})

my_submission["Id"] =  my_submission["Id"].astype(int)
# you could use any filename. We choose submission here
my_submission.to_csv('submission_RF.csv', index=False)



import xgboost as XGB

the_model = XGB.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=4, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, random_state =7, nthread = -1)


the_model.fit(X, y)

y_predict = the_model.predict(test_df.drop("Id", axis = 1))
y_predict

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': np.floor(np.expm1(y_predict))})

my_submission["Id"] =  my_submission["Id"].astype(int)
# you could use any filename. We choose submission here
my_submission.to_csv('submission_Boost_4.csv', index=False)


