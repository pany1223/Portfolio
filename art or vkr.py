# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 23:30:25 2022

@author: Павел
"""
import pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import math

data = pd.read_excel('bsgrusm7.xlsx')

data = data[['IDBOOK', 'IDSCHOOL','IDCLASS', 'IDSTUD', 'BSBG01', 'BSBGSLM', 'BSBGSVM', 'BSBGSCM', 'BSBGHER', 'BSDGHER', 'BSBGSB', 'BSBG07','BSBG13B','BSBG03','BSMIBM01','BSMIBM02','BSMIBM03','BSMIBM04','BSMIBM05']]
data['mark'] = (data['BSMIBM01'] + data['BSMIBM02'] + data['BSMIBM03'] + data['BSMIBM04'] + data['BSMIBM05'])/5
data.info()
data.describe()
data.nunique()
data.drop(['BSMIBM01','BSMIBM02','BSMIBM03','BSMIBM04','BSMIBM05'], axis = 1, inplace = True)

teachers = pd.read_excel("btmrusm7.xlsx")
teachers = teachers[['IDTEALIN','IDSCHOOL', 'IDTEACH', 'BTBG06C','BTBGTJS','BTBG07B', 'BTDGTJS']]

school = pyreadstat.read_sav("bcgrusm7.sav")[0]
school = school[['IDSCHOOL', 'BCBG03A', 'BCBG03B', 'BCBG04', 'BCBG05B', 'BCBGDAS', 'BCDGSBC','BCDGEAS']]

data = data.drop([3901, 3902,3903,3904])
data.isna().sum()

data.columns


for i in range(1,len(data.columns)):
    data.iloc[:,i] = data.iloc[:,i].map(lambda a: float("NaN") if (a == 9 or a == 999999) else a)
data.isna().sum()

for i in range(1,len(school.columns)):
    school.iloc[:,i] = school.iloc[:,i].map(lambda a: float("NaN") if (a == 9 or a == 999999) else a)
school.isna().sum()

for i in range(1,len(teachers.columns)):
    teachers.iloc[:,i] = teachers.iloc[:,i].map(lambda a: float("NaN") if (a == 9 or a == 999999) else a)
teachers.isna().sum()

for i in data.columns[1:]:
    print(data[data[i]==9][i].count())
msno.matrix(data.sort_values('BSBG01'))
msno.bar(data)
msno.dendrogram(data)

data.BSBG07.hist()

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

mice_imputer = IterativeImputer()
data['BSBG07'] = mice_imputer.fit_transform(data[['BSBG07']])
data['BSBG07'].isnull().sum()
data['BSBG07'].hist()
data.isna().sum()

data[['BSBG01', 'BSBGSLM', 'BSBGSVM', 'BSBGSCM', 'BSBGHER', 'BSDGHER', 'BSBGSB', 'BSBG13B', 'BSBG03']] = mice_imputer.fit_transform(data[['BSBG01', 'BSBGSLM', 'BSBGSVM', 'BSBGSCM', 'BSBGHER', 'BSDGHER', 'BSBGSB', 'BSBG13B', 'BSBG03']])
data['BSBG01'] = data['BSBG01'].apply(lambda z: math.floor(z))
data['BSDGHER'] = data['BSDGHER'].apply(lambda z: math.floor(z))
data['BSBG13B'] = data['BSBG13B'].apply(lambda z: np.round(z, 0))
data['BSBG07'] = data['BSBG07'].apply(lambda z: np.round(z, 0))

school.isna().sum()
teachers.isna().sum()


school['BCBG05B'] = mice_imputer.fit_transform(school[['BCBG05B']])
teachers['BTBG06C'] = mice_imputer.fit_transform(teachers[['BTBG06C']])


sns.boxplot(x = "BSBG01", y = 'mark', data = data, hue = "BSDGHER")
sns.boxplot(x = "BSBG07", y = 'mark', data = data, hue = "BSDGHER")
sns.boxplot(x = "BSBG07", y = 'mark', data = data, hue = "BSBG01")

sns.countplot(x = "BSBG13B", data = data, hue = 'BSDGHER')
sns.scatterplot(x = 'BSBGSB', y = 'BSBGHER', hue = "BSBG01", data = data)

sns.jointplot(x = 'BSBGSB', y = 'BSBGSLM', data = data, )


data['BSBG01'].value_counts()
data['BSBG13B'].value_counts()
sns.heatmap(data.iloc[:,4:].corr(method = 'spearman'))

corr = data.iloc[:,4:].corr(method = 'spearman')

stu_teach = pd.merge(data, teachers,  how='inner', left_on=['IDCLASS'], right_on = ["IDTEACH"])
stu_teach.isna().sum()

stu_teach_school = pd.merge(stu_teach, school,  how='inner', left_on=['IDSCHOOL_x'], right_on = ["IDSCHOOL"])
stu_teach_school.isna().sum()
stu_teach_school.drop(['IDTEALIN', 'IDSCHOOL_y', 'IDTEACH', 'IDSCHOOL_x'],axis = 1,  inplace=True)

stu_teach_left = pd.merge(data, teachers,  how='left', left_on=['IDCLASS'], right_on = ["IDTEACH"])
msno.matrix(stu_teach_left.sort_values("IDCLASS"))

from statsmodels.formula.api import ols


rich = stu_teach_school.sort_values("BSBGHER")[(stu_teach_school.index > np.percentile(stu_teach_school.index, 75))]
rich['dummy_rich'] = 1

poor = stu_teach_school.sort_values("BSBGHER")[(stu_teach_school.index < np.percentile(stu_teach_school.index, 25))]
poor['dummy_poor'] = 1


stu_teach_school = stu_teach_school.merge(poor[['IDSTUD', 'dummy_poor']], how = 'left')
stu_teach_school = stu_teach_school.merge(rich[['IDSTUD', 'dummy_rich']], how = 'left')
stu_teach_school = stu_teach_school.fillna(0)

stu_teach_school['BSBG13B'] = stu_teach_school['BSBG13B']*(-1) + 5
stu_teach_school['BTBG07B'] = stu_teach_school['BTBG07B'] *(-1) + 5

stu_teach_school['diff'] = stu_teach_school['BTBG07B'] - stu_teach_school['BSBG13B']
model = ols('mark ~ BSBG01*dummy_poor + BSBG01*dummy_rich+ BSBGSLM*dummy_rich + BSBGSLM*dummy_poor + BSBGSCM*dummy_rich + BSBGSCM*dummy_poor + BSBG07*dummy_poor + BSBG07*dummy_rich+ BTDGTJS* dummy_rich+ BTDGTJS* dummy_poor + BTBG06C*dummy_poor + BTBG06C*dummy_rich + BCDGSBC* dummy_poor + BCDGSBC* dummy_rich + BCBG04* dummy_poor + BCBG04* dummy_rich  + BSBG13B* dummy_poor + BSBG13B* dummy_rich + diff * dummy_poor + diff * dummy_rich' , data = stu_teach_school)
result = model.fit()
result.summary()
stu_teach_school[stu_teach_school['dummy_poor'] == 1]['diff'].mean()
stu_teach_school[stu_teach_school['dummy_poor'] == 1]['diff'].median()

stu_teach_school[(stu_teach_school['dummy_rich'] == 0) & (stu_teach_school['dummy_poor'] == 0)]['diff'].mean()
stu_teach_school[(stu_teach_school['dummy_rich'] == 0) & (stu_teach_school['dummy_poor'] == 0)]['diff'].mean()

stu_teach_school["SES"] = 0
stu_teach_school['dummy_rich'] = stu_teach_school['dummy_rich'].apply(lambda x: x+2 if x == 1 else 0)
stu_teach_school["SES"] = stu_teach_school['dummy_rich'] 
stu_teach_school['dummy_rich'] = stu_teach_school['dummy_rich'].apply(lambda x: x-2 if x == 3 else 0)
stu_teach_school["SES"] =stu_teach_school["dummy_poor"] + stu_teach_school["SES"]
stu_teach_school["SES"] = stu_teach_school["SES"].apply(lambda x: x+2 if x == 0 else x)
sns.countplot(data = stu_teach_school, x = 'diff', hue = "SES" )
stu_teach_school.groupby("SES")['diff'].agg(['mean', 'median'])
#Пошёл RANDOM FOREST 
rich = stu_teach_school.sort_values("BSBGHER")[(stu_teach_school.index > np.percentile(stu_teach_school.index, 75))]
rich['dummy_rich'] = 1
poor = stu_teach_school.sort_values("BSBGHER")[(stu_teach_school.index < np.percentile(stu_teach_school.index, 25))]
best_mark = stu_teach_school.sort_values("mark")[(stu_teach_school.index > np.percentile(stu_teach_school.index, 75))]
best_mark['mark'].min()
best_mark['mark'].max()
poor['resil'] = poor.mark.apply(lambda x: 0 if x < best_mark['mark'].min() else 1)
poor[]

from statsmodels.compat import lzip
import statsmodels.stats.api as sms

#perform Bresuch-Pagan test
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(result.resid, result.model.exog)

lzip(names, test)

from scipy import stats
import numpy as np

# Calculate the z-scores
z_scores = stats.zscore(stu_teach_school[['BSBG01','BSBGSLM','BSBGSCM','BSBG07','BTDGTJS','BTBG06C','BCDGSBC','BCBG04', 'BSBG13B','diff']])
z_scores

# Convert to absolute values
abs_z_scores = np.abs(z_scores)

# Select data points with a z-scores above or below 3
filtered_entries = (abs_z_scores >= 3).all(axis=1)

# Filter the dataset
df_wo_outliers = stu_teach_school[filtered_entries]
df_wo_outliers.shape

z = np.abs(stats.zscore(stu_teach_school[['BSBG01','BSBGSLM','BSBGSCM','BSBG07','BTDGTJS','BTBG06C','BCDGSBC','BCBG04', 'BSBG13B','diff']]))
print(np.where(z > 3))
np.where(z > 3)[0]
stu_teach_school.drop(np.where(z > 3)[0], inplace = True)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
X = poor.drop(['resil', 'IDBOOK', 'IDSCHOOL', 'IDCLASS', 'IDSTUD', 'mark', 'BSBGHER'], axis = 1)
y = poor.resil

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=17)
clf_rf = RandomForestClassifier(random_state=0)
parametrs = {'n_estimators': [i for i in range(10,51, 10)], 'max_depth': [i for i in range(1,12, 2)], "min_samples_leaf" : [i for i in range(1,8)], 'min_samples_split':[i for i in range(2,10, 2)]}
grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=3)
grid_search_cv_clf.fit(X_train, y_train)
grid_search_cv_clf.score(X_train, y_train)

grid_search_cv_clf.fit(X_test, y_test)
grid_search_cv_clf.score(X_test, y_test)

feature_importance = grid_search_cv_clf.best_estimator_.feature_importances_
df = pd.DataFrame({'features': list(X), 'feature_importance' : feature_importance})
aa = df.sort_values('feature_importance',ascending = False)

grid_search_cv_clf.predict_proba(X)


from sklearn.metrics import roc_auc_score,roc_curve
roc_auc_score(y, grid_search_cv_clf.predict_proba(X.values)[:,1])


X = poor.drop(['resil', 'IDBOOK', 'IDSCHOOL', 'IDCLASS', 'IDSTUD', 'mark', 'BSBGHER'], axis = 1)
y = poor.resil

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=17)
clf_rf = RandomForestClassifier(random_state=0)
parametrs = {'n_estimators': [i for i in range(10,51, 10)], 'max_depth': [i for i in range(1,12, 2)], "min_samples_leaf" : [i for i in range(1,8)], 'min_samples_split':[i for i in range(2,10, 2)]}
grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=3)
grid_search_cv_clf.fit(X_train, y_train)
grid_search_cv_clf.score(X_train, y_train)

grid_search_cv_clf.fit(X_test, y_test)
grid_search_cv_clf.score(X_test, y_test)

feature_importance = grid_search_cv_clf.best_estimator_.feature_importances_
df = pd.DataFrame({'features': list(X), 'feature_importance' : feature_importance})
aa = df.sort_values('feature_importance',ascending = False)

grid_search_cv_clf.predict_proba(X)


from sklearn.metrics import roc_auc_score,roc_curve
roc_auc_score(y, grid_search_cv_clf.predict_proba(X.values)[:,1])



#Кластеризация
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kmeans_interp.kmeans_feature_imp import KMeansInterp

new_df = stu_teach_school.iloc[:,3:].drop("IDSCHOOL", axis = 1)

scaled_df = StandardScaler().fit_transform(new_df)

kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


kmeans = KMeans(init="random", n_clusters=4, n_init=10, random_state=1)

#fit k-means algorithm to data
kmeans.fit(scaled_df)

#view cluster assignments for each observation
kmeans.labels_


new_df['cluster'] = kmeans.labels_

new_df['cluster'].hist()
sns.scatterplot(data = stu_teach_school, x = 'mark', y = 'BSBGHER', hue = 'cluster')

sns.pairplot(new_df,hue = "cluster")

sns.boxplot(data = new_df)


X = new_df.groupby("cluster").agg({"mean", "median", "std"})

