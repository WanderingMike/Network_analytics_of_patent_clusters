# Regression Problem
# importsimport numpy as npimport pandas as pdimport matplotlib.pyplot as pltimport seaborn as seimport warningsfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import r2_score, mean_absolute_error, mean_squared_errorfrom sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
# Data Fetchdf='DATAFRAME_OBJECT'df.head()
# Selected Columnsfeatures=['PK', 'SK', 'TS', 'PCD', '560', 'output']target='forward_citations'
# X & YX=df[features]Y=df[target]
# Data Cleaningdef NullClearner(value):	if(isinstance(value, pd.Series) and (value.dtype in ['float64','int64'])):		value.fillna(value.mean(),inplace=True)		return value	elif(isinstance(value, pd.Series)):		value.fillna(value.mode()[0],inplace=True)		return value	else:return valuex=X.columns.to_list()for i in x:	X[i]=NullClearner(X[i])Y=NullClearner(Y)
# Handling AlphaNumeric FeaturesX=pd.get_dummies(X)

# Correlation Matrix
f,ax = plt.subplots(figsize=(18, 18))matrix = np.triu(X.corr())se.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)plt.show()
# Data split for training and testingX_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
#Model Parametersparam={'loss': 'least_squares', 'learning_rate': 0.05494327645607078, 'max_iter': 3011, 'max_depth': 3, 'tol': 0.03941857933397222}
# Model Initializationmodel=HistGradientBoostingRegressor(**param)model.fit(X_train,Y_train)
# Metrics
y_pred=model.predict(X_test)print('R2 Score: {:.2f}'.format(r2_score(Y_test,y_pred)))print('Mean Absolute Error {:.2f}'.format(mean_absolute_error(Y_test,y_pred)))print('Mean Squared Error {:.2f}'.format(mean_squared_error(Y_test,y_pred)))