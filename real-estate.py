#!/usr/bin/env python
# coding: utf-8

# ## price predictor
# 

# In[1]:


import pandas as pd


# In[2]:


housing =pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS']


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))


# In[ ]:





# ## Train-Test splitting

# In[10]:


import numpy as np
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]


# In[11]:


#train_set,test_set=split_train_test(housing,0.2)


# In[12]:


#print(f"rows in train: {len(train_set)}\nrows in test: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f"rows in train: {len(train_set)}\nrows in test: {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]



# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# ## correlations

# In[17]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


from pandas.plotting import scatter_matrix
attr = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attr],figsize=(12,8))


# In[19]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=1)


# In[20]:


housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## trying attribute combinations

# In[21]:


# housing["TAXRM"]=housing['TAX']/housing['RM']
# housing.head()


# In[22]:


corr_matrix = housing.corr()
#corr_matrix['MEDV'].sort_values(ascending=False)


# # scikit-learn design

# primarly, three types of objects
# 1. Estimators - it estimates some parameter based on dataset. eg: imputer it has a fit method and transform method. Fit method - fits the dataset and calculates internal parameters
# 
# 2. Transformers - takes input and returns output based on the learning from fit(). it also has a convenience function called fit_transform() which fits and then transforms.
# 
# 3. Predictors - LinearRegression model is an example of predictor.  fit() and predit() are two common functions. it also gives score function which will evaluate the predictions.

# ## feature scaling

# primarily,two types of features scaling methods:
# 1. min-max scaling (normalizaion)
#     (value-min)/(max-min)
#     sklearn provides a class called MinMaxScaler for this 
#     
# 2. standardization
#     (value-mean)/std
#     sklearn provides a class called StandardScaler for this

# ## creating a pipeline

# In[23]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler())
])


# In[24]:


housing_num = my_pipeline.fit_transform(housing)


# In[ ]:





# In[25]:


housing.shape


# ## selecting model

# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num,housing_labels)


# In[27]:


some_data = housing.iloc[:5]


# In[28]:


some_lables = housing_labels.iloc[:5]


# In[29]:


prepared_data = my_pipeline.transform(some_data)


# In[30]:


model.predict(prepared_data)


# In[31]:


list(some_lables)


# ## Evaluating the model
# 
# 
# 
# 

# In[32]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[33]:


rmse


# ## using better evaluation technique - cross validation

# In[34]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)


# In[35]:


rmse_scores


# In[36]:


def print_scores(scores):
    print("scores:",scores)
    print("mean:",scores.mean())
    print("std dev:",scores.std())


# In[37]:


print_scores(rmse_scores)


# ## saving the model

# In[38]:


from joblib import dump,load
dump(model,'dragon.joblib')


# ## testing 

# In[43]:


X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions,list(Y_test))


# In[41]:


final_rmse


# In[44]:


prepared_data[0]


# ## using model

# In[47]:


from joblib import dump,load
import numpy as np
model = load('dragon.joblib')
input = np.array([[0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98]])
model.predict(input)


# In[ ]:




