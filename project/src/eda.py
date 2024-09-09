#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

# %%
# Importando os dados

df = pd.read_csv('../data/soil_measures.csv')
df
# %%
df.describe().T
# %%
df.isna().sum().sort_values()
# %%
df['crop'].value_counts()
# %%
