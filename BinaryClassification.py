#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# ### Load The Data
# 
# The first step we take is to load the binary dataset into the program.

# In[92]:


import pathlib
import pandas as pd
import numpy as np
import argparse
from settings import *

def load_binary_data():
    print('Loading binary class data in note...')
    X_train_path = pathlib.Path().joinpath(BINARY_DIR, 'X_train.csv')
    X_train_df = pd.read_csv(X_train_path, header=None)
    X_train_df.info()

    y_train_path = pathlib.Path().joinpath(BINARY_DIR, 'Y_train.csv')
    y_train_df = pd.read_csv(y_train_path, header=None)
    print('Unique values', y_train_df.iloc[:, 0].unique())

    X_test_path = pathlib.Path().joinpath(BINARY_DIR, 'X_test.csv')
    X_test_df = pd.read_csv(X_test_path, header=None)
    X_test_df.info()

    data_dict = {
        'X_train': X_train_df,
        'X_test': X_test_df,
        'y_train': y_train_df
    }
    print('\n')

    return data_dict


data_dict = load_binary_data()


# ### Clean The Data
# 
# We clean the data by removing the features taken from a random normal distribution.

# In[93]:


def clean_data(data_dict):
    print('Cleaning data...')
    X_train_df = data_dict['X_train']
    X_test_df = data_dict['X_test']
    y_train_df = data_dict['y_train']

    X_train_df = X_train_df.drop(X_train_df.columns[900:916], axis=1)  # df.columns is zero-based pd.Index
    X_test_df = X_test_df.drop(X_test_df.columns[900:916], axis=1)  # df.columns is zero-based pd.Index

    print('X train after cleaning:')
    X_train_df.info()
    print('X_test after cleaning')
    X_test_df.info()
    data_dict = {
        'X_train': X_train_df,
        'X_test': X_test_df,
        'y_train': y_train_df
    }
    print('\n')
    return data_dict


data_dict = clean_data(data_dict)


# ### Visualise Data
# 
# We visualise the target frequencies to see whether the dataset is imbalanced.

# In[94]:


import seaborn as sns
import matplotlib.pyplot as plt

def plot_target_frequency(data_dict):
    y_train = data_dict['y_train']
    y_train.columns = ['label']
    print(y_train.columns)
    total = float(len(y_train))
    print('total', total)
    plot = sns.countplot(x='label', data=y_train)
    for p in plot.patches:
        height = p.get_height()
        plot.text(p.get_x() + p.get_width() / 2.,
                height + 3,
                '{:1.3f}'.format(height / total),
                ha="center")
    plt.show()
    
plot_target_frequency(data_dict)


# ### Split training dataset to have a mock test set
# 
# We split the dataset in order to obtain a mock test set, since we don't have the target labes that correspond to the data in the X_test.csv file.

# In[95]:


from sklearn.model_selection import train_test_split


# In[96]:


X_train, X_val, y_train, y_val = train_test_split(data_dict['X_train'], data_dict['y_train'], test_size=0.2, random_state=23, shuffle=True, stratify=data_dict['y_train'])


# ### Obtain undersampled dataset
# 
# Undersample the data that  will be used for  training. We do not undersample the mock testing set as we want to keep the distribution of the classes close to the distribution of the original dataset.

# In[97]:


from imblearn.under_sampling import RandomUnderSampler


# In[98]:


rus = RandomUnderSampler(random_state=0)


# In[99]:


X_train_under, y_train_under = rus.fit_resample(X_train, y_train)


# In[100]:


data_dict_under = {
    'y_train': y_train_under
}
plot_target_frequency(data_dict_under)


# ### Prepare Inputs 
# 
# We convert the dataframes into numpy ndarrays.

# In[101]:


X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
y_train = y_train.to_numpy().ravel()
y_val = y_val.to_numpy().ravel()


# In[102]:


X_train_under = X_train_under.to_numpy()
y_train_under = y_train_under.to_numpy().ravel()


# # Machine  Learning

# ### Create Scorer
# 
# We define the scoring metrics to be used during cross-validation.

# In[103]:


scorer = ['balanced_accuracy', 'accuracy']


# ### Training KNN
# 
# Here we define our estimator as well as the grid search. The estimator is a pipeline in which the features are scaled, PCA is perfromed and a KNN classifier is trained. 

# In[104]:


import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV


# In[105]:


def kn_cross_validate_pca(X_train, y_train, scorer):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')), ('cf', KNeighborsClassifier())])

    params = {
        'cf__n_neighbors': [3],
        'cf__leaf_size': [30],
        'cf__p': [2] 
    }
    
    print('Grid: ', params)
    
    print('Scorer: ', scorer)
    
    cf = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])

    start = time.time()
    cf.fit(X_train, y_train)
    end = time.time()
    print('K-nearest neighbors cross-val time elapsed: ', end - start)

    print('Best params: ', cf.best_params_)    

    print('PCA number of components', cf.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = cf.best_score_ * 100
    
    acc_score =  cf.cv_results_['mean_test_accuracy'][cf.best_index_] * 100

    print("Best cross-val balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-val accuracy score: " + str(round(acc_score, 2)) + '%')
    
    cv_results = pd.DataFrame(cf.cv_results_)
    display(cv_results)

    print('\n')
    return cf.best_estimator_


# In[106]:


knn_cf = kn_cross_validate_pca(X_train, y_train, scorer)


# ### Making predictions on the validation set with KNN
# 
# Make predictions on the validation set and evaluate the classifier's performance.

# In[175]:


y_pred = knn_cf.predict(X_val)


# ### Confusion matrix for KNN
# The confusion matrix will tell us how each class was misclassified. 

# In[108]:


from sklearn.metrics import confusion_matrix


# In[109]:


cf = confusion_matrix(y_val, y_pred)


# In[110]:


import seaborn as sns
import numpy as np


# In[111]:


# sns.heatmap(cf, annot=True, fmt='g', cmap='Blues')
sns.heatmap(cf/np.sum(cf), annot=True, fmt='.2%', cmap='Blues')


# ### Classification report for KNN
# The classification report will give us detailed evaluation metrics.

# In[112]:


from sklearn.metrics import classification_report


# In[113]:


cr = classification_report(y_val, y_pred)


# In[114]:


print(cr)


# In[179]:


from sklearn.metrics import balanced_accuracy_score

print('Balanced accuracy: ', balanced_accuracy_score(y_val, y_pred))


# ### Train KNN with undersampled dataset
# 
# Here, we train the KNN classifier using the undersampled dataset.
# 

# In[119]:


knn_cf_under = kn_cross_validate_pca(X_train_under, y_train_under, scorer=scorer)


# ### Making predictions on evaluation set with KNN trained on the undersampled dataset
# We use the KNN classifier trained using the undersampled dataset to make predictions on the validation set to allow  us to evaluate its performance.

# In[172]:


y_pred_under = knn_cf_under.predict(X_val)


# ### Confusion matrix for KNN trained on the  undersampled dataset
# The confusion matrix will tell us how each class was misclassified. 

# In[173]:


cf_under = confusion_matrix(y_val, y_pred_under)


# In[174]:


# sns.heatmap(cf_under, annot=True, fmt='g', cmap='Blues')
sns.heatmap(cf_under/np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')


# ### Classification report for KNN trained on the undersampled dataset
# The classification report will give us detailed evaluation metrics.

# In[123]:


cr_under = classification_report(y_val, y_pred_under)


# In[124]:


print(cr_under)


# ### Training Random Forest  Classifier
# 
# Here we define our estimator as well as the grid search. The estimator is a pipeline in which the features are scaled, PCA is perfromed and a RandomForest classifier is trained. 

# In[125]:


from sklearn.ensemble import RandomForestClassifier


# In[126]:


def rf_cross_validate_pca(X_train, y_train, scorer):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')), ('cf', RandomForestClassifier())])

    params = {
        'cf__n_estimators': [400],
        'cf__max_depth': [None],
        'cf__min_samples_split': [2],
    }
    
    print('Grid: ', params)
    
    print('Scorer: ', scorer)
    
    cf = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])

    start = time.time()
    cf.fit(X_train, y_train)
    end = time.time()
    print('Random Forest cross-val time elapsed: ', end - start)

    print('Best params: ', cf.best_params_)    

    print('PCA number of components', cf.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = cf.best_score_ * 100
    
    acc_score =  cf.cv_results_['mean_test_accuracy'][cf.best_index_] * 100

    print("Best cross-val balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-val accuracy score: " + str(round(acc_score, 2)) + '%')
    
    cv_results = pd.DataFrame(cf.cv_results_)
    display(cv_results)

    print('\n')
    return cf.best_estimator_


# In[127]:


rf_cf = rf_cross_validate_pca(X_train, y_train, scorer=scorer)


# ### Making predictions on the validation set with RF
# We use the RF classifier to make predictions on the validation set to allow  us to evaluate its performance.### Making predictions on the validation set

# In[180]:


y_pred = rf_cf.predict(X_val)


# ### Confusion matrix for RF
# The confusion matrix will tell us how each class was misclassified.

# In[129]:


from sklearn.metrics import confusion_matrix


# In[130]:


cf = confusion_matrix(y_val, y_pred)


# In[131]:


import seaborn as sns
import numpy as np


# In[132]:


# sns.heatmap(cf, annot=True, fmt='g', cmap='Blues')
sns.heatmap(cf/np.sum(cf), annot=True, fmt='.2%', cmap='Blues')


# ### Classification report for RF
# The classification report will give us detailed evaluation metrics.

# In[133]:


from sklearn.metrics import classification_report


# In[134]:


cr = classification_report(y_val, y_pred)


# In[135]:


print(cr)


# In[181]:


from sklearn.metrics import balanced_accuracy_score

print('Balanced accuracy: ', balanced_accuracy_score(y_val, y_pred))


# ### Train RF with undersampled dataset
# Here, we train the RandomForest classifier using the undersampled dataset.

# In[140]:


rf_cf_under = rf_cross_validate_pca(X_train_under, y_train_under, scorer=scorer)


# ### Making prediction on evaluation set with RF trained on undersampled dataset
# We use the RF classifier trained using the undersampled dataset to make predictions on the validation set to allow  us to evaluate its performance.

# In[169]:


y_pred_under = rf_cf_under.predict(X_val)


# ### Confusion matrix for RF trained on the undersampled dataset
# The confusion matrix will tell us how each class was misclassified. 

# In[170]:


cf_under = confusion_matrix(y_val, y_pred_under)


# In[171]:


# sns.heatmap(cf_under, annot=True, fmt='g', cmap='Blues')
sns.heatmap(cf_under/np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')


# ### Classification report for RF trained on the undersampled dataset
# The classification report will give us detailed evaluation metrics.

# In[144]:


cr_under = classification_report(y_val, y_pred_under)


# In[145]:


print(cr_under)


# ### Training SVM Classifier
# 
# Here we define our estimator as well as the grid search. The estimator is a pipeline in which the features are scaled, PCA is perfromed and an SVM classifier is trained. 

# In[146]:


from sklearn.svm import LinearSVC


# In[147]:


def svc_cross_validate_pca(X_train, y_train, scorer):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline(
        [('sc', StandardScaler()), ('pca', PCA(n_components=0.99, svd_solver='full')), ('cf', LinearSVC())])

    params = {
        'cf__C': [1.0],
        'cf__loss': ['squared_hinge'],
        'cf__dual': [False],
    }
    
    print('Grid: ', params)
    
    print('Scorer: ', scorer)
    
    cf = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring=scorer, refit=scorer[0])

    start = time.time()
    cf.fit(X_train, y_train)
    end = time.time()
    print('SVC cross-val time elapsed: ', end - start)

    print('Best params: ', cf.best_params_)    

    print('PCA number of components', cf.best_estimator_.named_steps['pca'].n_components_)

    balanced_acc_score = cf.best_score_ * 100
    
    acc_score =  cf.cv_results_['mean_test_accuracy'][cf.best_index_] * 100

    print("Best cross-val balanced accuracy score: " + str(round(balanced_acc_score, 2)) + '%')
    print("Best cross-val accuracy score: " + str(round(acc_score, 2)) + '%')
    
    cv_results = pd.DataFrame(cf.cv_results_)
    display(cv_results)

    print('\n')
    return cf.best_estimator_


# In[148]:


svc_cf = svc_cross_validate_pca(X_train, y_train, scorer=scorer)


# ### Making predictions on the validation set with SVC
# We use the SVC classifier to make predictions on the validation set to allow  us to evaluate its performance.

# In[182]:


y_pred = svc_cf.predict(X_val)


# ### Confusion matrix for SVC
# The confusion matrix will tell us how each class was misclassified.

# In[150]:


from sklearn.metrics import confusion_matrix


# In[151]:


cf = confusion_matrix(y_val, y_pred)


# In[152]:


import seaborn as sns
import numpy as np


# In[153]:


sns.heatmap(cf/np.sum(cf), annot=True, fmt='.2%', cmap='Blues')


# In[154]:


from sklearn.metrics import classification_report


# In[155]:


cr = classification_report(y_val, y_pred)


# In[156]:


print(cr)


# In[183]:


from sklearn.metrics import balanced_accuracy_score

print('Balanced accuracy: ', balanced_accuracy_score(y_val, y_pred))


# ### Train SVC with undersampled dataset
# Here, we train the SVC classifier using the undersampled dataset.

# In[162]:


svc_cf_under = svc_cross_validate_pca(X_train_under, y_train_under, scorer=scorer)


# ### Making prediction on evaluation set with SVC under
# We use the SVC classifier trained using the undersampled dataset to make predictions on the validation set to allow  us to evaluate its performance.

# In[163]:


y_pred_under = svc_cf_under.predict(X_val)


# ### Confusion matrix for SVC trained on the undersampled dataset
# The confusion matrix will tell us how each class was misclassified. 

# In[164]:


cf_under = confusion_matrix(y_val, y_pred_under)


# In[165]:


# sns.heatmap(cf_under, annot=True, fmt='g', cmap='Blues')
sns.heatmap(cf_under/np.sum(cf_under), annot=True, fmt='.2%', cmap='Blues')


# ### Classification report for SVC trained on the undersampled dataset
# The classification report will give us detailed evaluation metrics.

# In[166]:


from sklearn.metrics import classification_report


# In[167]:


cr_under = classification_report(y_val, y_pred_under)


# In[168]:


print(cr_under)


# ## Produce the Y_test.csv file
# 
# This is the file that will be used to evaluate the final performance of the classifier. After analysing the results we decided that we will be using the SVM classifier that was trained on the original dataset. 

# In[187]:


X_test = data_dict['X_test']


# In[188]:


y_test = svc_cf.predict(X_test)


# In[193]:


output_path = pathlib.Path().joinpath(BINARY_DIR, 'Y_test.csv')
np.savetxt(output_path, y_test, fmt="%s")


# In[ ]:




