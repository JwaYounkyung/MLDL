import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_x = pd.read_csv('/Users/jwa/Desktop/ds_gist/ca3-JwaYounkyung-master/JwaYounkyung/preprocess/train_preprocess_1.csv', header=0)
#X_train = df_x.values

df_y = pd.read_csv('/Users/jwa/Desktop/ds_gist/ca3-JwaYounkyung-master/JwaYounkyung/preprocess/train_label_1.csv', header=0)
#y_train = df_y.values
df_y_time = df_y.iloc[:,1]
df_y_spent = df_y.iloc[:,2]

df_merge_1 = pd.concat([df_y_time,df_x], axis=1)
#df_merge_1 = pd.concat([df_y_spent,df_x], axis=1)

correlation = df_merge_1.corr(method='pearson')
columns = correlation.nlargest(10, 'survival_time').index
#columns = correlation.nlargest(10, 'amount_spent').index

plt.figure(figsize=(12,12))
plt.title('Correlation by survival_time', fontsize=14)

correlation_map = np.corrcoef(df_merge_1[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

#heat_4 의 현재 축을 확인
#heat_4 = sns.heatmap(df.corr(), cmap='Blues', linewidths=0.5, vmax=0.5,  fmt = '.2f' , annot=True)
bottom, top = heatmap.get_ylim()
print("bottome :", bottom, "top :", top)

# y축 범위 조정
heatmap.set_ylim(bottom + 0.5, top - 0.5)

plt.show()