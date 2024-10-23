import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

X = pd.read_csv('train.csv')
X.dropna(subset = ['SalePrice'],axis = 0,inplace = True)
y = X.SalePrice
X.drop(['SalePrice'],axis = 1,inplace = True)
cols_with_missing = [col for col in X.columns
                    if X[col].isnull().any()]
X.drop(cols_with_missing,axis = 1)
train_X,valid_X,train_y,valid_y = train_test_split(X,y,random_state = 21,train_size = 0.8,test_size = 0.2)

def score_dataset(train_X,valid_X,train_y,valid_y) :
  model = RandomForestRegressor(n_estimators = 200,random_state = 21)
  model.fit(train_X,train_y)
  predictions = model.predict(valid_X)
  mae = mean_absolute_error(valid_y,predictions)
  return mae

object_cols = [col for col in train_X.columns
              if train_X[col].dtype == 'object']
good_label_cols = [col for col in object_cols
                  if set(valid_X[col]).issubset(set(train_X[col]))]
bad_label_cols = list(set(object_cols) - set(good_label_cols))


label_train_X = train_X.drop(bad_label_cols , axis = 1)
label_valid_X = valid_X.drop(bad_label_cols , axis = 1)


ordinal_encoder = OrdinalEncoder()
label_train_X[good_label_cols] = ordinal_encoder.fit_transform(train_X[good_label_cols])
label_valid_X[good_label_cols] = ordinal_encoder.transform(valid_X[good_label_cols])

print(score_dataset(label_train_X,label_valid_X,train_y,valid_y))
