import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = pd.read_csv('train.csv')
X.dropna(subset = ['SalePrice'],axis = 0, inplace = True)
y = X.SalePrice
X.drop(['SalePrice'], axis = 1 , inplace = True )

cols_with_missing = [col for col in X.columns
                    if X[col].isnull().any()]
X.drop(cols_with_missing,axis = 1 , inplace = True)
train_X,val_X,train_y,valid_y = train_test_split(X , y , random_state = 21 , train_size = 0.8 , test_size = 0.2)

def score_dataset(train_X,val_X,train_y,valid_y) :
  model = RandomForestRegressor(n_estimators = 200 , random_state = 21)
  model.fit(train_X,train_y)
  predictions = model.predict(val_X)
  mae = mean_absolute_error(valid_y,predictions)
  return mae

object_cols = [col for col in train_X.columns
              if train_X[col].dtype == 'object']

low_cardinality_cols = [cname for cname in object_cols
                       if train_X[cname].nunique() < 10 and train_X[cname].dtype == 'object']
OH_encoder = OneHotEncoder(handle_unknown = 'ignore' , sparse_output = False)
OH_train_cols = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_valid_cols = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))

OH_train_cols.index = train_X.index
OH_valid_cols.index = val_X.index

num_train_cols = train_X.drop(object_cols , axis = 1)
num_valid_cols = val_X.drop(object_cols , axis = 1)

OH_train_X = pd.concat([num_train_cols,OH_train_cols] , axis = 1)
OH_val_X = pd.concat([num_valid_cols,OH_valid_cols] , axis = 1)

OH_train_X.columns = OH_train_X.columns.astype(str)
OH_val_X.columns = OH_val_X.columns.astype(str)

print(score_dataset(OH_train_X,OH_val_X,train_y,valid_y))
