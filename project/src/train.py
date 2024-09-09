#%%

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# %%
#Lendo os dados 

df = pd.read_csv('../data/soil_measures.csv')
# %%
target_cat = pd.Categorical(df['crop'])
target_encoded = target_cat.codes
target_categories = target_cat.categories
df['crop'] = target_encoded
# %%
features = [column for column in df.columns if column != 'crop' ]
target = 'crop'
# %%
X_train,X_test,y_train,y_test = train_test_split(df[features]
                                                 ,df[target]
                                                 ,test_size=0.3
                                                 ,stratify=df[target])
# %%
steps = [('scaller',StandardScaler())
         ,('model',DecisionTreeClassifier(max_depth=12,random_state=42))]

pip_model = Pipeline(steps)
# %%
pip_model.fit(X_train,y_train)
# %%
pip_model.score(X_train,y_train)
# %%
predictions_train = pip_model.predict(X_train)
predictions_test = pip_model.predict(X_test)
#%%
print(classification_report(y_test,predictions_test))
# %%
pip_model['model'].feature_importances_
# %%
