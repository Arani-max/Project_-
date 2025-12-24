# %% [markdown]
# # Анализ сна и образа жизни
# 
# ## Загрузка данных

# %%
import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')
print("Данные загружены. Размер:", df.shape)