#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Library

# In[98]:


import warnings
warnings.filterwarnings('ignore')


# In[99]:


import pandas as pd
import numpy as np


# In[100]:


from sklearn.ensemble import RandomForestRegressor
#from sklearn.neighbors import NearestNeighbors


# In[101]:


from sklearn.ensemble import RandomForestClassifier


# In[102]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[103]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[104]:


from sklearn.metrics import r2_score


# In[105]:


from sklearn.metrics import mean_squared_error


# In[127]:


import statsmodels.api as sm


# In[128]:


from sklearn.model_selection import RandomizedSearchCV


# In[129]:


from sklearn.model_selection import GridSearchCV


# In[130]:


from pprint import pprint
from IPython import get_ipython


# In[131]:




# In[132]:




# In[133]:





# In[ ]:





# ---------
# ## Data

# #### Train, Test Data

# In[139]:


trdf=pd.read_csv(r'C:\Users\ashut\Desktop\BTP\website\03_Train_Data.csv', index_col=0)


# In[142]:


tedf = pd.read_csv(r'C:\Users\ashut\Desktop\BTP\website\03_Test_Data.csv', index_col=0)


# In[143]:


trdf.set_index('Lateral_ID (i)', inplace = True)


# In[144]:


tedf.set_index('Lateral_ID (i)', inplace = True)


# In[145]:


trdf.sort_index(inplace = True)


# In[146]:


tedf.sort_index(inplace = True)


# In[147]:


trdf


# In[148]:


tedf


# In[149]:


tedf.columns


# In[150]:


X_tr = np.array(trdf[['~y(1)(i)', '~y(2)(i)', '~y(3)(i)', '~y(4)(i)']])


# In[151]:


X_tr.shape


# In[152]:


y_tr = np.array(trdf[[  'Affected Customers s(i)', 'Lateral Anomaly Status x(i)']])


# In[153]:


y_tr.shape


# In[154]:


X_te = np.array(tedf[['~y(1)(i)', '~y(2)(i)', '~y(3)(i)', '~y(4)(i)']])


# In[155]:


X_te.shape


# In[156]:


y_te = np.array(tedf[['Affected Customers s(i)', 'Lateral Anomaly Status x(i)']])


# In[157]:


y_te.shape


# In[ ]:





# In[158]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# --------------
# --------------
# ## 1) Base Model : RF Regressor
# - x(i)
# - s(i)

# In[159]:

## use an odd number of trees to prevent predictions of 0.5
rf = RandomForestRegressor(n_estimators=11)


# In[160]:


rf.fit(X_tr, y_tr)


# In[161]:


y_pred = rf.predict(X_te)


# In[162]:




# all predictions will be continous so just round the continous ones
y_pred[:, 1] = y_pred[:, 1].round()


# ## Results:

# In[163]:


cm = confusion_matrix(y_te[:,1], y_pred[:,1], labels=[0,1,2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['N', 'L1', 'L2'])
disp.plot()


# In[164]:


accuracy_score(y_te[:,1], y_pred[:,1]) 


# In[165]:


f1_score(y_te[:,1], y_pred[:,1], average='weighted') 


# In[166]:


precision_score(y_te[:,1], y_pred[:,1], average='weighted') 


# In[167]:


recall_score(y_te[:,1], y_pred[:,1], average='weighted') 


# In[168]:


r2_score(y_te[:,0], y_pred[:,0])


# In[169]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te[:,0],
    'predicted s(i)': y_pred[:,0],
    'true x(i)': y_te[:,1],
    'predicted x(i)': y_pred[:,1]
})


# In[170]:


dfff.set_index('ID', inplace = True)


# In[171]:


dfff.hist()


# --------------
# --------------
# ## 2) Model : improving s(i) through x(i) 
# - RF classifier, x(i)
# - RF regressor, s(i)

# In[172]:


rf2 = RandomForestClassifier()


# In[173]:


rf2.fit(X_tr, y_tr[:,1])


# In[174]:


y_pred21 = rf2.predict(X_te)


# ## Results:

# In[175]:


cm = confusion_matrix(y_te[:,1], y_pred21, labels=[0,1,2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['N', 'L1', 'L2'])
disp.plot()


# In[176]:


accuracy_score(y_te[:,1], y_pred21) 


# In[177]:


f1_score(y_te[:,1], y_pred21, average='weighted') 


# In[178]:


precision_score(y_te[:,1], y_pred21, average='weighted') 


# In[179]:


recall_score(y_te[:,1], y_pred21, average='weighted') 


# #### Training s(i) using original s(i), testing s(i) using predicted s(i)

# In[180]:


X_tr2 = np.array(trdf[['~y(1)(i)', '~y(2)(i)', '~y(3)(i)', '~y(4)(i)', 'Lateral Anomaly Status x(i)']])


# In[181]:


y_tr2 = np.array(trdf['Affected Customers s(i)'])


# In[182]:


y_te2 = np.array(tedf['Affected Customers s(i)'])


# In[183]:


X_te2 = tedf[['~y(1)(i)', '~y(2)(i)', '~y(3)(i)', '~y(4)(i)']]


# In[184]:


X_te2['predicted x(i)'] = y_pred21


# In[185]:


X_te2 = np.array(X_te2)


# In[186]:


# use an odd number of trees to prevent predictions of 0.5
rf22 = RandomForestRegressor(n_estimators=11)


# In[187]:


rf22.fit(X_tr2, y_tr2)


# In[188]:


y_pred22 = rf22.predict(X_te2)


# In[189]:


r2_score(y_te2, y_pred22)


# In[190]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te2,
    'predicted s(i)': y_pred22,
})


# In[191]:


dfff.set_index('ID', inplace = True)


# In[192]:


dfff.hist()


# slightly improved

# #### Training s(i) using x(i): with cross validation

# In[196]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[197]:


rf_random.fit(X_tr2, y_tr2)


# In[198]:


rf_random.best_params_


# In[199]:


best_random = rf_random.best_estimator_


# In[200]:


y_predr = best_random.predict(X_te2)


# In[201]:


r2_score(y_te2, y_predr)


# In[202]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te2,
    'predicted s(i)': y_predr,
})


# In[203]:


dfff.set_index('ID', inplace = True)


# In[204]:


dfff.hist()


# - - - 

# #### Training s(i) independently

# In[205]:


# use an odd number of trees to prevent predictions of 0.5
rf23 = RandomForestRegressor(n_estimators=11)


# In[206]:


rf23.fit(X_tr, y_tr[:,0])


# In[207]:


y_pred23 = rf23.predict(X_te)


# In[208]:


r2_score(y_te[:,0], y_pred23)


# In[209]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te[:,0],
    'predicted s(i)': y_pred23,
})


# In[210]:


dfff.set_index('ID', inplace = True)


# In[211]:


dfff.hist()


# #### Training s(i) independently: with cross validation

# In[212]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[213]:


rf_random.fit(X_tr, y_tr[:,0])


# In[214]:


rf_random.best_params_


# In[215]:


best_random = rf_random.best_estimator_


# In[216]:


y_predr = best_random.predict(X_te)


# In[217]:


r2_score(y_te[:,0], y_predr)


# In[218]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te[:,0],
    'predicted s(i)': y_predr,
})


# In[219]:


dfff.set_index('ID', inplace = True)


# In[220]:


dfff.hist()



# --------------
# --------------
# ## 2.2) Cross validated Model : RF Classifier
# - x(i)

# In[221]:


rf2 = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[222]:


rf_random.fit(X_tr, y_tr[:,1])

# In[223]:


y_pred21 = rf_random.predict(X_te)


# ## New code Added

# In[ ]:


def pre_x(y1, y2, y3, y4):
  test_new = [y1,y2,y3,y4]
  y_pred21 = rf_random.predict(test_new)  ##predicted x(i)  
  return y_pred21


# ## Results:

# In[224]:


cm = confusion_matrix(y_te[:,1], y_pred21, labels=[0,1,2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['N', 'L1', 'L2'])
disp.plot()


# In[225]:


accuracy_score(y_te[:,1], y_pred21) 


# In[226]:


f1_score(y_te[:,1], y_pred21, average='weighted') 


# In[227]:


precision_score(y_te[:,1], y_pred21, average='weighted') 


# In[228]:


recall_score(y_te[:,1], y_pred21, average='weighted') 


# In[229]:


df_x = pd.DataFrame({
    'IDs': tedf.index,
    'True x(i)': y_te[:,1],
    'Predicted x(i)': y_pred21
})


# In[230]:


df_x


# In[232]:


#df_x.to_csv('C:\Users\ashut\Desktop\BTP\website\04 model_i_model_2_2_x_i.csv')


# --------------
# --------------
# ## 3) Model : predicting s(i) with normalized counts
# - RF regressor, s(i)

# In[233]:


trdf['N~y(1)(i)'] = trdf['~y(1)(i)']/ trdf['N(i)']


# In[234]:


trdf['N~y(2)(i)'] = trdf['~y(2)(i)']/ trdf['N(i)']
trdf['N~y(3)(i)'] = trdf['~y(3)(i)']/ trdf['N(i)']
trdf['N~y(4)(i)'] = trdf['~y(4)(i)']/ trdf['N(i)']


# In[235]:


X_tr3 = np.array(trdf[['N~y(1)(i)', 'N~y(2)(i)', 'N~y(3)(i)', 'N~y(4)(i)']])


# In[236]:


X_tr3.shape


# In[237]:


y_tr = np.array(trdf[[  'Affected Customers s(i)', 'Lateral Anomaly Status x(i)']])


# In[238]:


y_tr.shape


# In[239]:


tedf['N~y(1)(i)'] = tedf['~y(1)(i)']/ tedf['N(i)']


# In[240]:


tedf['N~y(2)(i)'] = tedf['~y(2)(i)']/ tedf['N(i)']
tedf['N~y(3)(i)'] = tedf['~y(3)(i)']/ tedf['N(i)']
tedf['N~y(4)(i)'] = tedf['~y(4)(i)']/ tedf['N(i)']


# In[241]:


X_te3 = np.array(tedf[['N~y(1)(i)', 'N~y(2)(i)', 'N~y(3)(i)', 'N~y(4)(i)']])


# In[242]:


X_te3.shape


# In[243]:


# use an odd number of trees to prevent predictions of 0.5
rf3 = RandomForestRegressor(n_estimators=11)


# In[244]:


rf3.fit(X_tr3, y_tr[:,0])


# ## New code Added
# 

# In[ ]:


def pre_s(y1,y2,y3,y4,N):
  tedf['N~y(1)(i)'] = y1/N
  tedf['N~y(2)(i)'] = y2/N
  tedf['N~y(3)(i)'] = y3/N
  tedf['N~y(4)(i)'] = y4/N

  X_te3 = np.array(tedf[['N~y(1)(i)', 'N~y(2)(i)', 'N~y(3)(i)', 'N~y(4)(i)']])
  y_pred3 = rf3.predict(X_te3)  ##predicted s(i)
  return y_pred3


# In[245]:


y_pred3 = rf3.predict(X_te3)


# In[246]:


r2_score(y_te[:,0], y_pred3)


# In[247]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te[:,0],
    'predicted s(i)': y_pred3,
})


# In[248]:


dfff.set_index('ID', inplace = True)


# In[249]:


dfff.hist()


# In[250]:


mean_squared_error(y_te[:,0], y_pred3)


# In[251]:


#rmse
mean_squared_error(y_te[:,0], y_pred3, squared = False)


# ## Cross-validated

# In[252]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[253]:


rf_random.fit(X_tr3, y_tr[:,0])


# In[254]:


rf_random.best_params_


# In[255]:


best_random = rf_random.best_estimator_


# In[256]:


y_predr3 = best_random.predict(X_te3)




# In[257]:


r2_score(y_te[:,0], y_predr3)


# In[258]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te[:,0],
    'predicted s(i)': y_predr3,
})


# In[259]:


dfff.set_index('ID', inplace = True)


# In[260]:


dfff.hist()


# In[261]:


mean_squared_error(y_te[:,0], y_predr3)


# In[262]:


#rmse
mean_squared_error(y_te[:,0], y_predr3, squared = False)

# In[265]:


dfff.to_csv(r'C:\Users\ashut\Desktop\BTP\website\04 model_i_model_3_s_i.csv')


# In[ ]:





# In[ ]:





# --------------
# --------------
# ## 3.2) Model : predicting s(i) with normalized counts using x(i)'s
# - RF regressor, s(i)
# - Use true x(i) as an additional feature for training data.
# - Use predicted x(i) as an additional feature on test data

# In[266]:


X_tr32 = np.array(trdf[['N~y(1)(i)', 'N~y(2)(i)', 'N~y(3)(i)', 'N~y(4)(i)', 'Lateral Anomaly Status x(i)']])


# In[267]:


X_tr32.shape


# In[268]:


y_tr32 = np.array(trdf[ 'Affected Customers s(i)'])


# In[269]:


y_tr32


# In[270]:


y_te32 = np.array(tedf[ 'Affected Customers s(i)'])


# In[271]:


X_te32 = tedf[['N~y(1)(i)', 'N~y(2)(i)', 'N~y(3)(i)', 'N~y(4)(i)']]


# In[272]:


X_te32['Predicted Lateral Anomaly Status x(i)'] = y_pred21


# In[273]:


X_te32


# In[274]:


# use an odd number of trees to prevent predictions of 0.5
rf32 = RandomForestRegressor(n_estimators=11)


# In[275]:


rf32.fit(X_tr32, y_tr32)


# In[276]:


y_pred32 = rf32.predict(X_te32)


# In[277]:


r2_score(y_te32, y_pred32)


# In[278]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te32,
    'predicted s(i)': y_pred32,
})


# In[279]:


dfff.set_index('ID', inplace = True)


# In[280]:


dfff.hist()


# In[281]:


mean_squared_error(y_te32, y_pred32)


# In[282]:


#rmse
mean_squared_error(y_te32, y_pred32, squared = False)


# ## Cross-validated

# In[283]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[284]:


get_ipython().run_cell_magic('time', '', '# Fit the random search model\nrf_random.fit(X_tr32, y_tr32)')


# In[285]:


rf_random.best_params_


# In[286]:


best_random = rf_random.best_estimator_


# In[287]:


y_predr32 = best_random.predict(X_te32)


# In[288]:


r2_score(y_te32, y_predr32)


# In[289]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te32,
    'predicted s(i)': y_predr32,
})


# In[290]:


dfff.set_index('ID', inplace = True)


# In[291]:


dfff.hist()


# In[292]:


mean_squared_error(y_te32, y_predr32)


# In[293]:


#rmse
mean_squared_error(y_te32, y_predr32, squared = False)


# - # Keep this on hold!
# - # Use grid search cv

# In[ ]:





# In[ ]:





# --------------
# --------------
# ## 4) Model : Poisson Modelling for s(i)

# In[294]:


X_tr1_2 =   sm.add_constant(X_tr)


# In[295]:


np.shape(X_tr1_2)


# In[296]:


poisson_training_results = sm.GLM(y_tr[:,0], X_tr1_2, family=sm.families.Poisson()).fit()


# In[297]:


print(poisson_training_results.summary())


# In[298]:


X_te1_2 =   sm.add_constant(X_te)


# In[299]:


poisson_predictions = poisson_training_results.get_prediction(X_te1_2)


# In[300]:


#summary_frame() returns a pandas DataFrame
predictions_summary_frame = poisson_predictions.summary_frame()


# In[301]:


predictions_summary_frame


# In[302]:


r2_score(y_te[:,0], predictions_summary_frame['mean']) 


# In[303]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    'true s(i)': y_te[:,0],
    'predicted s(i)': predictions_summary_frame['mean']
})


# In[304]:


dfff.set_index('ID', inplace = True)


# In[305]:


dfff.hist()


# In[306]:


r2_score(dfff['true s(i)'], dfff['predicted s(i)'])


# In[ ]:





# ## Negative Binomial Modelling:
# - sample the normalized values

# In[307]:


nb_training_results = sm.GLM(y_tr[:,0], X_tr1_2, family=sm.families.NegativeBinomial()).fit()


# In[308]:


print(nb_training_results.summary())


# In[309]:


nb_predictions = nb_training_results.get_prediction(X_te1_2)


# In[310]:


#summary_frame() returns a pandas DataFrame
predictions_summary_frame = nb_predictions.summary_frame()


# In[311]:


predictions_summary_frame


# In[312]:


r2_score(y_te[:,0], predictions_summary_frame['mean'])


# In[313]:


dfff = pd.DataFrame({
    'ID': tedf.index,
    's(i)': y_te[:,0],
    'predicted s(i)': predictions_summary_frame['mean']
})


# In[314]:


dfff.set_index('ID', inplace = True)


# In[315]:


dfff.hist()


# In[316]:


r2_score(dfff['s(i)'], dfff['predicted s(i)'])


# In[ ]:





# In[ ]:





# In[ ]:





# ## Conclusions:
# 
# - x(i): Use random search, cross-validated, RF classifier on original ~y(i)(j)'s
# - s(i): Use random search, cross-validated, RF regressor on normalised N~y(i)(j)'s

# ## Try:
# - NN

# In[ ]:





# In[ ]:





# In[ ]:




