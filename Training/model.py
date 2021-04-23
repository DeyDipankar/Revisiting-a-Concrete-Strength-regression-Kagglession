#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.feature_selection import SelectFromModel
from feature_engine.selection import SelectByShuffling,SmartCorrelatedSelection
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,ElasticNet


# In[3]:


data = pd.read_csv('./Concrete_Data_Yeh.csv')
data.head()


# ## EDA

# In[4]:


print(len(data))
print(data.isnull().sum())
print(data.info())


# In[26]:


plt.figure(figsize = (10,6))
sns.heatmap(data.corr(method = 'pearson'),annot = True)


# In[6]:


data.hist(figsize = (10,8))


# In[7]:


data.describe()


# ## Train Test Split

# In[8]:


X = data.iloc[:,:-1]
y= data.iloc[:,-1]
print(X.columns)
print(y)


# In[48]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)


# ## Feature Scaling

# In[49]:


sc = StandardScaler()
X_train_transformed = sc.fit_transform(X_train)
X_test_transformed = sc.transform(X_test)
X_train_transformed = pd.DataFrame(X_train_transformed, columns = X_train.columns)
X_test_transformed = pd.DataFrame(X_test_transformed, columns = X_train.columns)


# ## Feature Selection

# In[50]:


# USing scikitlearn's selectfrommodel embedded method
sel = SelectFromModel(RandomForestRegressor())
sel.fit(X_train,y_train)
selected_features = X_train.columns[sel.get_support()]
pd.Series(sel.estimator_.feature_importances_.ravel(),index = X_train.columns).sort_values(ascending=False).plot(kind = 'bar')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
print('Selected features based on importance :', selected_features)


# In[51]:


# Using SelectByShuffling hybrid method from feature_engine
rf= RandomForestRegressor()
sel = SelectByShuffling(
        variables = None,
        estimator = rf,
        scoring = 'neg_root_mean_squared_error',
        cv = 3,
        random_state = 0
)
sel.fit(X_train,y_train)
sel.features_to_drop_


# As we can see from above,based on neg_root_mean_squared_error metric, we're left with 'cement', and 'age' as the most important features,this means that these are the most important predictors rather than the other features. Since there are less number of features,I'll try to keep all the features provided no multicollinearity is present.

# In[52]:


# finding out correlated features with feature-engines SmartCorrelatedSelection
sel = SmartCorrelatedSelection(
            variables= None,
            method = 'pearson',
            missing_values= 'raise',
            selection_method = 'variance',
            estimator = None,
            scoring = 'neg_root_mean_squared_error',
            cv = 3,
            threshold = 0.65
)
sel.fit(X_train,y_train)
sel.features_to_drop_


# Although there's a correlation between 'water' and 'superplasticizer', but correlation 0.66 < 0.7. So, even we keep this,it will not be a strong multicollinearity. But, here we'll drop ['superplasticizer']

# In[53]:


X_train = X_train.drop(['superplasticizer'],axis = 1)
X_test =X_test.drop(['superplasticizer'],axis = 1)
print(X_train.columns)
print(X_test.columns)


# ## Choosing a baseline model

# In[46]:


lr = LinearRegression()
laso = Lasso()
rdg = Ridge()
sgd = SGDRegressor(penalty = None, eta0 = 0.000001)
elstnet = ElasticNet()
poly_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree = 2, include_bias = False )),
    ("lin_reg", LinearRegression())
])
rf = RandomForestRegressor()
estimators = [lr,laso,rdg,sgd,elstnet,poly_reg,rf]
for estimator in estimators:
    score = cross_val_score(estimator,X_train,y_train,scoring = 'neg_root_mean_squared_error', cv = 5)
    print(estimator,score.mean(),score.std())


# ## Hyperparameter Optimization

# In[56]:


result = []
for deg in [2,3,5,7,10]:
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree = deg, include_bias = False )),
        ("lin_reg", LinearRegression())
    ])
    score = cross_val_score(model,X_train,y_train,scoring = 'neg_root_mean_squared_error', cv = 5)
    result.append((deg,score.mean(),score.std()))
result


# We can see from above that the loss is minimum if we choose degree 3 polynomial

# In[63]:


regressor = model = Pipeline([
        ("poly_features", PolynomialFeatures(degree = 3, include_bias = False )),
        ("lin_reg", LinearRegression())
    ])


# ## Evaluating model on test data

# In[64]:


regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
loss = mean_squared_error(y_test,y_pred)
print(loss)


# Difference between R-squared and adjusted R-square:
# 
# 1. Adjusted R-square can be negative only when R-square is very close to zero.
# 
# 2. Adjusted R-Square values are always less than or equal to R-square but never be greater than R-Square.
# 
# 3. If we add an independent variable to a model every time then, the R-squared increases, despite the independent variable being insignificant. This means that the R-square value increases when an independent variable is added despite its significance. Adjusted R-squared increases only when the independent variable is significant and affects the dependent variable.

# In[65]:


R = r2_score(y_test,y_pred)
print(R)


# In[66]:


n = len(X_test)
p = len(X_test.columns)
adjusted_r2 = 1 - (1- R**2) * (n - 1)/(n - p - 1)
print(adjusted_r2)


# In[67]:


plt.scatter(y_pred,y_test)
plt.xlabel('y_pred')
plt.ylabel('y_test')


# ## Save model

# In[72]:


with open('model','wb') as f:
    pickle.dump(model,f)

