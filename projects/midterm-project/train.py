import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# DATA IMPORT
df = pd.read_csv('./data/clean_data.csv')

df_train = df.drop('msuccess', axis=1).reset_index(drop=True)
y_train = df.msuccess.reset_index(drop=True)

# FEATURE PREPARATION
def encode_categorical_vars(df: pd.DataFrame, dv: DictVectorizer = None) -> pd.DataFrame:
  df_dicts = df.to_dict(orient='records')

  if not dv:
    dv = DictVectorizer(sparse=False)
    dv.fit(df_dicts)
  
  df_encoded = pd.DataFrame(
    data=dv.transform(df_dicts),
    columns=dv.get_feature_names_out()
  )

  return df_encoded, dv

X_train, dv = encode_categorical_vars(df_train)

# MODEL TRAINING
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42)
model.fit(X_train, y_train)

# Saving the model
output_file = f'model_decisiontree.bin'
with open(output_file, 'wb') as f_out:
  pickle.dump((dv, model), f_out)
print(f"The model was saved to {output_file}")