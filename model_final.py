import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier


trdf=pd.read_csv(r'03_Train_Data.csv', index_col=0)

tedf = pd.read_csv(r'03_Test_Data.csv', index_col=0)

trdf.set_index('Lateral_ID (i)', inplace = True)

tedf.set_index('Lateral_ID (i)', inplace = True)

trdf.sort_index(inplace = True)

tedf.sort_index(inplace = True)

##for x

X_tr1 = np.array(trdf[['~y(1)(i)', '~y(2)(i)', '~y(3)(i)', '~y(4)(i)']])
y_tr = np.array(trdf[[  'Affected Customers s(i)', 'Lateral Anomaly Status x(i)']])


rf1 = RandomForestClassifier()

rf1.fit(X_tr1, y_tr[:,1])

def pre_x(y1, y2, y3, y4):
  x_te1 = [y1,y2,y3,y4]
  y_pred1 = rf1.predict([x_te1])  ##predicted x(i)  
  return y_pred1


##For s

trdf['N~y(1)(i)'] = trdf['~y(1)(i)']/ trdf['N(i)']
trdf['N~y(2)(i)'] = trdf['~y(2)(i)']/ trdf['N(i)']
trdf['N~y(3)(i)'] = trdf['~y(3)(i)']/ trdf['N(i)']
trdf['N~y(4)(i)'] = trdf['~y(4)(i)']/ trdf['N(i)']

X_tr2 = np.array(trdf[['N~y(1)(i)', 'N~y(2)(i)', 'N~y(3)(i)', 'N~y(4)(i)']])

rf2 = RandomForestRegressor(n_estimators=1000, min_samples_split=10,min_samples_leaf=2,max_features='sqrt',max_depth=10,bootstrap=True)

rf2.fit(X_tr2, y_tr[:,0])

def pre_s(y1,y2,y3,y4,N):
  a = y1/N
  b = y2/N
  c = y3/N
  d = y4/N

  X_te2 = [a,b,c,d]
  y_pred2 = rf2.predict([X_te2])  ##predicted s(i)
  return y_pred2


##model k
dftr = pd.read_csv(r'05_Train_Data.csv', index_col =0)

dfte = pd.read_csv(r'05_Test_Data.csv', index_col =0)

X_tr3 = np.array(dftr.drop(['Customer_ID (k)', 'Customer Anomaly Status x(i)(k)'], axis=1))
y_tr2 = np.array(dftr['Customer Anomaly Status x(i)(k)'])

rf3 = RandomForestClassifier()
rf3.fit(X_tr3, y_tr2)

def pre_xik(l,N,x,s,y1k,y2k,y3k,y4k,y1,y2,y3,y4):
  X_te3 = [[l,N,x,s,y1k,y2k,y3k,y4k,y1,y2,y3,y4]]

  y_pred3 = rf3.predict(X_te3)
  return y_pred3

