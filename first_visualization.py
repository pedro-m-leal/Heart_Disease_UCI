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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

## Visualizations functions

def histograms(series):
    if isinstance(series,pd.Series):
        mn=int(min(series))
        mx=int(max(series))
        n_bins=len(range(mn,mx))+1
        ax=sns.distplot(series,n_bins,kde=False)
        ax.set_title('{}'.format(series.name))
        plt.show()
    elif isinstance(series,pd.DataFrame):
        cols=series.columns
        length=len(cols)
        if length%4==0:
            n_rows=4
            n_columns=length//4
        else:
            n_rows=4
            n_columns=length//4+1
        
        fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(length):
            mn=int(min(series[cols[i]]))
            mx=int(max(series[cols[i]]))
            n_bins=len(range(mn,mx))+1
            sns.distplot(series[cols[i]],n_bins,kde=False,ax=axis.flatten()[i])
            axis.flatten()[i].set_title(cols[i])
        plt.show()
    else:
        raise ValueError('Please use pd.Series or pd.DataFrame as argument.')
def boxplots(series):
    if isinstance(series,pd.Series):
        ax=sns.boxplot(series)
        ax.set_title('{}'.format(series.name))
        plt.show()
    elif isinstance(series,pd.DataFrame):
        cols=series.columns
        length=len(cols)
        if length%2==0:
            n_rows=2
            n_columns=length//2
        else:
            n_rows=2
            n_columns=length//2+1

        fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(length):
            sns.boxplot(series[cols[i]],ax=axis.flatten()[i])
            axis.flatten()[i].set_title(cols[i])
        plt.show()
    else:
        raise ValueError('Please use pd.Series or pd.DataFrame as argument.')
def countplots(series):
    if isinstance(series,pd.Series):
        ax=sns.countplot(series)
        ax.set_title('{}'.format(series.name))
        plt.show()
    elif isinstance(series,pd.DataFrame):
        cols=series.columns
        length=len(cols)
        if length%2==0:
            n_rows=2
            n_columns=length//2
        else:
            n_rows=2
            n_columns=length//2+1

        fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(length):
            sns.countplot(series[cols[i]],ax=axis.flatten()[i])
            axis.flatten()[i].set_title(cols[i])
        plt.show()
    else:
        raise ValueError('Please use pd.Series or pd.DataFrame as argument.')
def frequency_plot(df,feature,feature_to_group='LeagueId'):
    freq_prep=train_df[[feature_to_group,feature]].groupby([feature_to_group]).sum()
    counts=freq_prep[feature].values
    leagues=freq_prep.index.values
    length=len(leagues)
    if length%2==0:
        n_rows=2
        n_columns=length//2
    else:
        n_rows=2
        n_columns=length//2+1

    fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
    for i in range(len(leagues)):
        feature_counts=train_df[train_df[feature_to_group]==leagues[i]].groupby([feature]).count()[feature_to_group].values
        freqs=np.asarray(feature_counts)/sum(feature_counts)
        feature_values=train_df[train_df[feature_to_group]==leagues[i]].groupby([feature]).count()[feature_to_group].index.values
        sns.barplot(x=feature_values,y=freqs,ax=axis.flatten()[i])
        axis.flatten()[i].set_title('{} : {}'.format(feature_to_group,leagues[i]))
    plt.suptitle('{}'.format(feature))
    plt.show()

## Import Data

input=pd.read_csv('heart.csv')
df=input.copy(deep=True)

print('\n\n ============ Printing df head and create profile ============= \n\n')


print(df.head())

try:
    fn=open("df_pandas_profiling_before_processing.html")
    fn.close()
except:
    print('\n---- No profile exists yet. Creating profile. ----\n')
    profile = df.profile_report(title='Pandas Profiling Report')
    profile.to_file(output_file="df_pandas_profiling_before_processing.html")

numerical=['age','trestbps','chol','thalach','oldpeak']
binary=['sex','fbs','exang','target']
categorical=['cp','restecg','slope','ca','thal']

# Eliminate duplicate rows

print('\n\n ============ Eliminating duplicate rows ============= \n\n')

dups=df.duplicated()
df=df.iloc[dups[dups==False].index.tolist()]

df.reset_index(drop=True,inplace=True)

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



# Let's see how many outliers we have

print('\n\n ============ Detecting and removing outliers ============= \n\n')


df['outlier']=[0]*len(df['age'])

quantiles=df[numerical].quantile(q=[0.25,0.5,0.75]).transpose()
quantiles['iqr']=quantiles[0.75]-quantiles[0.25]
quantiles['min_lim']=quantiles[0.25]-1.5*quantiles['iqr']
quantiles['max_lim']=quantiles[0.75]+1.5*quantiles['iqr']



for column in numerical:
    outliers_up=((df[column]-quantiles['max_lim'].loc[column])>0)
    outliers_down=((df[column]-quantiles['min_lim'].loc[column])<0)
    df['outlier']=df['outlier']+outliers_up
    df['outlier']=df['outlier']+outliers_down



df_no_outlier=df[df['outlier']==0]

print(df_no_outlier.describe())


### Visualization before scaling

# Heatmap of correlations
colour=sns.diverging_palette(240,10,as_cmap=True)
sns.heatmap(df.corr(),annot=True,cmap=colour)
plt.title('Correlation Heatmap')
plt.show()

# Histograms of 'age'
fig, ax= plt.subplots(nrows=2,ncols=2)
count=0
for i in binary:
    axis=ax.flatten()[count]
    sns.distplot(df[df[i]==0]['age'],color='blue',label='Feature {} is 0.'.format(i),ax=ax.flatten()[count])
    sns.distplot(df[df[i]==1]['age'],color='red',label='Feature {} is 1.'.format(i),ax=ax.flatten()[count])
    axis.legend()
    plt.title('Age distribution in the dataset, separated by {}'.format(i))
    count+=1
plt.show()


# Countplots

sns.catplot(data=df,x='sex',col='target',hue='ca',kind='count')
plt.show()

# Boxenplots

sns.catplot(data=df,x='sex',y='chol',col='target',hue='ca',kind='boxen')
plt.show()

# Violin plots

sns.catplot(data=df,x='sex',y='chol',col='target',hue='ca',kind='violin')
plt.show()

# ECG with target

sns.catplot(data=df,x='target',hue='restecg',col='sex',kind='count')
plt.legend()
plt.suptitle("Rest ECG counts for 'sex' and 'target' features")
plt.show()


# Data Scaling -> Numeric features

print('\n\n ============ Scaling numeric features ============= \n\n')
rb=StandardScaler()
scaler=rb.fit(df_no_outlier[numerical])
df[numerical]=scaler.transform(df[numerical])


print(df[numerical].describe())

# Get dummies -> Categorical features

print('\n\n ============ Getting dummy variables for categorical features ============= \n\n')

df=df.join(pd.get_dummies(df[categorical],columns=categorical,drop_first=True))
df.drop(categorical,axis=1,inplace=True)
df.drop(['outlier'],axis=1,inplace=True)

print(df.head())

print(df.shape)

try:
    fn=open("df_pandas_profiling_after_processing.html")
    fn.close()
except:
    print('\n---- No profile exists yet. Creating profile. ----\n')
    profile = df.profile_report(title='Pandas Profiling Report')
    profile.to_file(output_file="df_pandas_profiling_after_processing.html")


# Heatmap of correlations
colour=sns.diverging_palette(240,10,as_cmap=True)
sns.heatmap(df.corr(),annot=True,cmap=colour)
plt.title('Correlation Heatmap after scaling and dummies')
plt.show()













    