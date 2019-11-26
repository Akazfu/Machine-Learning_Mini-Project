import pandas as pd
# import seaborn as sns

data = pd.read_csv('dataset/incomedata.csv', header=None)

print(data.head())
print(data.shape)

# sns.pairplot(data, x_vars=[0], y_vars = 14, height=7, aspect=0.7)
