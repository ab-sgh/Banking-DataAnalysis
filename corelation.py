#Banking Dataset Analysis

#Importing appropriate libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('banking_data.csv')
df['job'] = df['job'].astype('category').cat.codes
df['marital_status'] = df['marital_status'].astype('category').cat.codes
df['education'] = df['education'].astype('category').cat.codes
df['default'] = df['default'].astype('category').cat.codes
df['housing'] = df['housing'].astype('category').cat.codes
df['loan'] = df['loan'].astype('category').cat.codes
df['contact'] = df['contact'].astype('category').cat.codes
df['month'] = df['month'].astype('category').cat.codes
df['day_month'] = df['day_month'].astype('category').cat.codes
df['poutcome'] = df['poutcome'].astype('category').cat.codes
df['y'] = df['y'].astype('category').cat.codes


#Before analysis we should make the dataframe optimal
#Since marital and marital_status are alike, we will remove one
df.drop(columns='marital', inplace=True)

#both marital_status and education columns have three null entries. So, we will use deletion to delete these rows.
#Since the no. of faulty rows is very less compared to the total row size, deletion will not cause any bias
df.dropna(subset=['marital_status','education'], inplace = True)
corr_matrix= df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap= 'PuBuGn', fmt='.2f')
plt.show()