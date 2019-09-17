#Libraries for Data Visualization

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np 
import seaborn as sns
import time
import pandas_profiling as pp

from sklearn.preprocessing import StandardScaler, RobustScaler #Scaling Time and Amount
from mpl_toolkits import mplot3d
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.model_selection import StratifiedKFold, learning_curve, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import f1_score, precision_score, precision_recall_fscore_support, fbeta_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding, SpectralEmbedding, Isomap
import mca
import prince



## Import Data

input=pd.read_csv('heart.csv')
df=input.copy(deep=True)

print('\n\n ============ Printing df head ============= \n\n')


print(df.head())

numerical=['age','trestbps','chol','thalach','oldpeak']
binary=['sex','fbs','exang','target']
categorical=['cp','restecg','slope','ca','thal']




# Let us check for missing values

print('\n\n ============ Checking for missing values ============= \n\n')

x=sum(df.isnull().sum())
print('-> Is any value missing from the database?', '\n')
print('-> No! \n') if x==0 else print('-> Yes...', '\n') 

if x!=0:
    print('-> These are the rows with missing values:','\n')
    mv_rows=df[df.isnull().any(axis=1)]
    print(mv_rows)
    df.drop(mv_rows.index.tolist(),inplace=True)
    print('\n-> There should be no missing values now. \n')
    print(df.isnull().sum(), '\n \n \n')


# Eliminate duplicate rows

print('\n\n ============ Eliminating duplicate rows ============= \n\n')

dups=df.duplicated()
df=df.iloc[dups[dups==False].index.tolist()]

df.reset_index(drop=True,inplace=True)


# Descriptive statistics

print('\n\n ============ Descriptive statistics of the original dataset ============= \n\n')

print(df.describe())

too_much_chol=df[df['chol']>350]

# Change dtypes

df[numerical]=df[numerical].astype('float64')

# Removing almost empty categories

df['restecg']=df['restecg']-(df['restecg']==2)
df['ca']=df['ca']-(df['ca']==4)
df['thal']=df['thal']+(df['thal']==0)


# Let's split test and training sets

print('\n\n ============ Splitting test and training+cv sets ============= \n\n')

ss=ShuffleSplit(n_splits=1,test_size=0.3)
for train_index, test_index in ss.split(df):
    df_train_cv, df_test = df.loc[train_index], df.loc[test_index]

df_train_cv.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)



# Now we split in training and cross-validation sets

print('\n\n ============ Spliting training and cross-validation sets ============= \n\n')

ss2=ShuffleSplit(n_splits=1,test_size=0.4)

for train_index, cv_index in ss2.split(df_train_cv):
    df_train, df_cv = df_train_cv.loc[train_index], df_train_cv.loc[cv_index]



# Let's see how many outliers we have

print('\n\n ============ Detecting and removing outliers ============= \n\n')


df_train['outlier']=[0]*len(df_train['age'])

quantiles=df_train[numerical].quantile(q=[0.25,0.5,0.75]).transpose()
quantiles['iqr']=quantiles[0.75]-quantiles[0.25]
quantiles['min_lim']=quantiles[0.25]-1.5*quantiles['iqr']
quantiles['max_lim']=quantiles[0.75]+1.5*quantiles['iqr']



for column in numerical:
    outliers_up=((df_train[column]-quantiles['max_lim'].loc[column])>0)
    outliers_down=((df_train[column]-quantiles['min_lim'].loc[column])<0)
    df_train['outlier']=df_train['outlier']+outliers_up
    df_train['outlier']=df_train['outlier']+outliers_down



df_train_no_outlier=df_train[df_train['outlier']==0]

print(df_train_no_outlier.describe())


# Data Scaling -> Numeric features

print('\n\n ============ Scaling numeric features ============= \n\n')
rb=StandardScaler()
scaler=rb.fit(df_train_no_outlier[numerical])
df_train[numerical]=scaler.transform(df_train[numerical])
df_cv[numerical]=scaler.transform(df_cv[numerical])
df_test[numerical]=scaler.transform(df_test[numerical])

print(df_train[numerical].describe())

# Get dummies -> Categorical features

print('\n\n ============ Getting dummy variables for categorical features ============= \n\n')

df_train=df_train.join(pd.get_dummies(df_train[categorical],columns=categorical,drop_first=True))
df_train.drop(categorical,axis=1,inplace=True)
df_train.drop(['outlier'],axis=1,inplace=True)

df_cv=df_cv.join(pd.get_dummies(df_cv[categorical],columns=categorical,drop_first=True))
df_cv.drop(categorical,axis=1,inplace=True)

df_test=df_test.join(pd.get_dummies(df_test[categorical],columns=categorical,drop_first=True))
df_test.drop(categorical,axis=1,inplace=True)



# Doing MCA and PCA

print('\n\n ============ Doing PCA and MCA ============= \n\n')

dummies=['sex','fbs','exang','cp_1', 'cp_2', 'cp_3', 'restecg_1', 'slope_1', 'slope_2', 'ca_1', 'ca_2', 'ca_3', 'thal_2', 'thal_3']
numerical=['age','trestbps','chol','thalach','oldpeak']
all_cols=['sex','fbs','exang','cp_1', 'cp_2', 'cp_3', 'restecg_1', 'slope_1', 'slope_2', 'ca_1', 'ca_2', 'ca_3', 'thal_2', 'thal_3','age','trestbps','chol','thalach','oldpeak']

pca=PCA(n_components=len(numerical))
pca.fit(df_train[numerical])
df_train[numerical]=pca.transform(df_train[numerical])
df_cv[numerical]=pca.transform(df_cv[numerical])
df_test[numerical]=pca.transform(df_test[numerical])
print('Explained variance in the numerical values: ',pca.explained_variance_ratio_)

mca=prince.MCA(n_components=len(dummies))
mca.fit(df_train[dummies])
df_train[dummies]=mca.transform(df_train[dummies])
df_cv[dummies]=mca.transform(df_cv[dummies])
df_test[dummies]=mca.transform(df_test[dummies])

pca2=PCA(n_components=len(df_train[all_cols].columns.tolist()))
pca2.fit(df_train[all_cols])
df_train[all_cols]=pca2.transform(df_train[all_cols])
df_cv[all_cols]=pca2.transform(df_cv[all_cols])
df_test[all_cols]=pca2.transform(df_test[all_cols])



print('\n\nExplained variance in the binary values: ', mca.explained_inertia_)

print('\nModified dataframe:')
print(df_train.head())

try:
    fn=open("df_pandas_profiling_after_pca.html")
    fn.close()
except:
    print('\n---- No profile exists yet. Creating profile. ----\n')
    profile = df_train.profile_report(title='Pandas Profiling Report')
    profile.to_file(output_file="df_pandas_profiling_after_pca.html")


# Checking for correlations between features again

print('\n\n ============ Checking correlations ============= \n\n')

colour=sns.diverging_palette(240,10,as_cmap=True)
sns.heatmap(df_train.corr(),annot=True,cmap=colour)
plt.title('Correlation Heatmap')
plt.show()

# Split X and y

print('\n\n ============ Splitting in X and y ============= \n\n')

X_train=df_train.drop(['target'],axis=1)
y_train=df_train['target']

X_cv=df_cv.drop(['target'],axis=1)
y_cv=df_cv['target']

print('\n\n',X_train.shape,'\n\n')
print('\n\n',X_cv.shape,'\n\n')



# Select best features

selector = RFECV(DecisionTreeClassifier(max_depth=2), step=1, cv=5)
selector.fit(X_train,y_train)
ranking=selector.ranking_
best_columns=[a*b for a,b in zip(X_train.columns.tolist(),ranking<30)]
best_columns=list(filter(None,best_columns))
print('-> The best columns are:',best_columns)

X_train=X_train[best_columns]
X_cv=X_cv[best_columns]

sns.scatterplot(data=df_train,x='cp_3',y='sex',hue='target')
plt.show()
## Model training

print('\n\n ============ Setting up and training the model ============= \n\n')

dt=LogisticRegression(C=1000,solver='lbfgs') #MLPClassifier(max_iter=1000) #AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1)) #LogisticRegression(solver='lbfgs',C=100) #GaussianNB() #SVC(C=100,gamma=1,kernel='rbf')
dt.fit(X_train,y_train)
train_score=dt.score(X_train,y_train)
cv_score=dt.score(X_cv,y_cv)
preds_cv=dt.predict(X_cv)
probs_cv=dt.predict_proba(X_cv)

print(train_score,'\n' ,cv_score)

cm=confusion_matrix(y_cv,preds_cv)
sns.heatmap(cm,annot=True,cmap='PuBu')
plt.show()

## Plot the learning curve for the best model

X_train_cv=pd.concat([X_train,X_cv],ignore_index=True)
y_train_cv=pd.concat([y_train,y_cv],ignore_index=True)

train_sizes, train_scores, test_scores = learning_curve(dt, X_train_cv, y_train_cv, cv=5, n_jobs=1,scoring='accuracy')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title('Learning Curves')
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid(b=True)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()

## Plot the roc curve for the best model


fpr, tpr, thresholds = roc_curve(y_cv,probs_cv[:,1],)

plt.figure()
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid(b=True)
plt.plot([0,1],[0,1],color='b',dashes=[2,2])
plt.plot(fpr,tpr,color='r',label='Area Under ROC curve = {}'.format(roc_auc_score(y_cv,probs_cv[:,1])))
plt.legend()
plt.show()

## Predict on the test_set


# X_test=df_test.drop(['target'],axis=1)
# y_test=df_test['target']

# X_test=X_test[best_columns]

# test_preds=dt.predict(X_test)
# cm_test=confusion_matrix(y_test,test_preds)
# sns.heatmap(cm_test,annot=True,cmap='PuBu')
# plt.show()
# test_score=dt.score(X_test,y_test)
# print(test_score)





















    