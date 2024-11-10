import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle

# model parameters
C = 1.0
n_splits = 5

output_file = f'model_C={C}.bin'

# Data Cleaning
def standardise_colnames(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df.columns = df.columns.str.lower().str.replace(' ', '_')
  return df

def standardise_str_cols(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()

  for col in df.columns:
    if is_object_dtype(df[col]):
      df[col] = df[col].str.lower().str.replace(' ', '_')
  
  return df


df = pd.read_csv('customer-churn.csv')
df = standardise_colnames(df)
df = standardise_str_cols(df)

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce').fillna(0)
df.churn = (df.churn == 'yes').astype('int')
df.drop('customerid', axis=1, inplace=True)

# Validation Framework
df_train_full, df_test, y_train_full, y_test = train_test_split(
  df.drop('churn', axis=1), df.churn, test_size=.2, random_state=1)
df_train, df_val, y_train, y_val = train_test_split(
  df_train_full, y_train_full, test_size=.25, random_state=1)

df_train_full.reset_index(drop=True, inplace=True)
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

y_train_full = y_train_full.values
y_train = y_train.values
y_val = y_val.values
y_test = y_test.values

# Feature Preparation
def encode_vars(df: pd.DataFrame, dv: DictVectorizer = None):
  df_dicts = df.to_dict(orient='records')

  if not dv:
    dv = DictVectorizer(sparse=False)
    dv.fit(df_dicts)
  
  encoded_data = dv.transform(df_dicts)

  return encoded_data, dv

X_train, dv = encode_vars(df_train)
X_val, _ = encode_vars(df_val, dv)
X_test, _ = encode_vars(df_test, dv)

# Training
def train(df: pd.DataFrame, y_train: np.array, C=1.0):
  X_train, dv = encode_vars(df)
  model = LogisticRegression(max_iter=10000, C=C)
  model.fit(X_train, y_train)

  return dv, model

def predict(df: pd.DataFrame, dv, model):
  X, _ = encode_vars(df, dv)
  y_pred = model.predict_proba(X)[:, 1]

  return y_pred

# cross-validation
print(f'Cross-validation with C={C}')
print("===========================")

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for i, (train_idx, val_idx) in enumerate(kfold.split(df_train_full)):
  df_train = df_train_full.iloc[train_idx]
  df_val = df_train_full.iloc[val_idx]

  y_train = y_train_full[train_idx]
  y_val = y_train_full[val_idx]

  dv, model = train(df_train, y_train, C=C)
  y_pred = predict(df_val, dv, model)

  auc = roc_auc_score(y_val, y_pred)
  print(f'AUC on fold {i}: {auc}')
  scores.append(auc)

print(f'C={C} AUC avg: {np.mean(scores):.3f} AUC std: {np.std(scores):.3f}')

print(f'Training final model with C={C}')
print("===============================")
dv, model = train(df_train_full, y_train_full, C=C)
y_pred = predict(df_test, dv, model)
print(y_pred)

# Saving the model
with open(output_file, 'wb') as f_out:
  pickle.dump((dv, model), f_out)
print(f"The model was saved to {output_file}")