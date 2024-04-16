<div style="background-color: #333; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);">
    <h1 style="text-align: center; font-size: 28px; color: #fff; margin-bottom: 10px;">Exploratory Data Analysis & Classification</h1>
    <p style="text-align: center; font-size: 16px; color: #fff;">This project investigates online food orders, employing analysis and visuals to grasp data nuances and offer strategic insights for businesses.</p>
</div>



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
```


```python
data = pd.read_csv('/kaggle/input/online-food-dataset/onlinefoods.csv')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Marital Status</th>
      <th>Occupation</th>
      <th>Monthly Income</th>
      <th>Educational Qualifications</th>
      <th>Family size</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Pin code</th>
      <th>Output</th>
      <th>Feedback</th>
      <th>Unnamed: 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>Female</td>
      <td>Single</td>
      <td>Student</td>
      <td>No Income</td>
      <td>Post Graduate</td>
      <td>4</td>
      <td>12.9766</td>
      <td>77.5993</td>
      <td>560001</td>
      <td>Yes</td>
      <td>Positive</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>Female</td>
      <td>Single</td>
      <td>Student</td>
      <td>Below Rs.10000</td>
      <td>Graduate</td>
      <td>3</td>
      <td>12.9770</td>
      <td>77.5773</td>
      <td>560009</td>
      <td>Yes</td>
      <td>Positive</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Male</td>
      <td>Single</td>
      <td>Student</td>
      <td>Below Rs.10000</td>
      <td>Post Graduate</td>
      <td>3</td>
      <td>12.9551</td>
      <td>77.6593</td>
      <td>560017</td>
      <td>Yes</td>
      <td>Negative</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>Female</td>
      <td>Single</td>
      <td>Student</td>
      <td>No Income</td>
      <td>Graduate</td>
      <td>6</td>
      <td>12.9473</td>
      <td>77.5616</td>
      <td>560019</td>
      <td>Yes</td>
      <td>Positive</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>Male</td>
      <td>Single</td>
      <td>Student</td>
      <td>Below Rs.10000</td>
      <td>Post Graduate</td>
      <td>4</td>
      <td>12.9850</td>
      <td>77.5533</td>
      <td>560010</td>
      <td>Yes</td>
      <td>Positive</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Family size</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>Pin code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>388.000000</td>
      <td>388.000000</td>
      <td>388.000000</td>
      <td>388.000000</td>
      <td>388.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.628866</td>
      <td>3.280928</td>
      <td>12.972058</td>
      <td>77.600160</td>
      <td>560040.113402</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.975593</td>
      <td>1.351025</td>
      <td>0.044489</td>
      <td>0.051354</td>
      <td>31.399609</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>12.865200</td>
      <td>77.484200</td>
      <td>560001.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>2.000000</td>
      <td>12.936900</td>
      <td>77.565275</td>
      <td>560010.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24.000000</td>
      <td>3.000000</td>
      <td>12.977000</td>
      <td>77.592100</td>
      <td>560033.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.000000</td>
      <td>4.000000</td>
      <td>12.997025</td>
      <td>77.630900</td>
      <td>560068.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.000000</td>
      <td>6.000000</td>
      <td>13.102000</td>
      <td>77.758200</td>
      <td>560109.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Checking for Missing Values and Duplicated Values**


```python
missing_values = data.isnull().sum()
duplicates = data.duplicated().sum()
missing_values, duplicates
```




    (Age                           0
     Gender                        0
     Marital Status                0
     Occupation                    0
     Monthly Income                0
     Educational Qualifications    0
     Family size                   0
     latitude                      0
     longitude                     0
     Pin code                      0
     Output                        0
     Feedback                      0
     Unnamed: 12                   0
     dtype: int64,
     103)



**Different Visualizations and plots**




```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(data['Age'], bins=15, kde=True, ax=ax[0], color='skyblue')
ax[0].set_title('Age Distribution')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Frequency')

sns.histplot(data['Family size'], bins=6, kde=True, ax=ax[1], color='salmon')
ax[1].set_title('Family Size Distribution')
ax[1].set_xlabel('Family Size')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

```


    
![png](output_8_0.png)
    


**The Countplots show the Gender, Martial Status, Monthly Income, and Education Qualification Distributions in the Dataset**


```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sns.countplot(x='Gender', data=data, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Gender Distribution')

sns.countplot(x='Marital Status', data=data, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Marital Status Distribution')

sns.countplot(x='Monthly Income', data=data, ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Monthly Income Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.countplot(x='Educational Qualifications', data=data, ax=axes[1, 1], color='orange')
axes[1, 1].set_title('Educational Qualifications Distribution')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

```


    
![png](output_10_0.png)
    


**Scatter Plot between Age and Family Size**


```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Family size', data=data, hue='Gender', style='Marital Status', s=100,
                palette='Set2')
plt.title('Relationship between Age and Family Size by Gender and Marital Status')
plt.xlabel('Age')
plt.ylabel('Family Size')
plt.legend(title='Gender / Marital Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```


    
![png](output_12_0.png)
    


**The chart shows how many orders are placed by people present in different income slabs**


```python
plt.figure(figsize=(12, 6))
sns.countplot(x='Monthly Income', hue='Output', data=data)
plt.title('Income Level vs. Online Ordering Behavior')
plt.xlabel('Monthly Income')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Ordered Online')
plt.tight_layout()
plt.show()
```


    
![png](output_14_0.png)
    



```python
plt.figure(figsize=(12, 6))
sns.countplot(x='Monthly Income', hue='Output', data=data, palette='dark')
plt.title('Income Level vs. Online Ordering Behavior')
plt.xlabel('Monthly Income')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Ordered Online')
plt.tight_layout()
plt.show()

```


    
![png](output_15_0.png)
    


**The charts show, the number of people present in different genders according to their feedbacks and also people with different educational qualifications according to their feedbacks**


```python
plt.figure(figsize=(12, 6))
sns.countplot(x='Feedback', hue='Gender', data=data, palette='pastel')
plt.title('Feedback by Gender')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(x='Feedback', hue='Educational Qualifications', data=data, palette='muted')
plt.title('Feedback by Educational Qualifications')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.legend(title='Educational Qualifications', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```


    
![png](output_17_0.png)
    



    
![png](output_17_1.png)
    


**The Box Plot shows the ordering behaviour of customers coming from different martial status and age**


```python
plt.figure(figsize=(12, 6))
sns.boxplot(x='Marital Status', y='Age', hue='Output', data=data , palette="dark")
plt.title('Ordering Behavior by Marital Status and Age')
plt.xlabel('Marital Status')
plt.ylabel('Age')
plt.legend(title='Ordered Online')
plt.tight_layout()
plt.show()

```


    
![png](output_19_0.png)
    


**The chart shows number of positive and negative feedbacks by people present in different income slabs**


```python
plt.figure(figsize=(14, 7))
sns.countplot(x='Monthly Income', hue='Feedback', data=data , palette="dark")
plt.title('Income Level and Feedback Sentiment')
plt.xlabel('Monthly Income')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Feedback', loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](output_21_0.png)
    


**The charts shows how many people from different education qualifications have ordered food online**


```python
plt.figure(figsize=(14, 7))
sns.countplot(x='Educational Qualifications', hue='Output', data=data , palette="dark")
plt.title('Educational Qualifications and Online Ordering')
plt.xlabel('Educational Qualifications')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Ordered Online', loc='upper right')
plt.tight_layout()
plt.show()
```


    
![png](output_23_0.png)
    


**Heatmap between different features present in the Dataset**


```python
correlation_matrix = data[['Age', 'Family size', 'latitude', 'longitude', 'Pin code']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=.5)
plt.title('Heatmap of Correlations Among Numerical Features')
plt.show()
```


    
![png](output_25_0.png)
    


**Violin Plot showing Monthly Income Vs. Age**


```python
plt.figure(figsize=(14, 8))
sns.violinplot(x='Monthly Income', y='Age', data=data , palette="dark")
plt.title('Violin Plots for Monthly Income vs. Age')
plt.xlabel('Monthly Income')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](output_27_0.png)
    


**Radar Chart showing relationships between Number of Positive Feedbacks, Number of Negative Feedbacks, Educational Qualifications, and Average Family Size**


```python
data = pd.read_csv('/kaggle/input/online-food-dataset/onlinefoods.csv')
data['Output_Numeric'] = data['Output'].map({'Yes': 1, 'No': 0})

data['Positive_Feedback'] = (data['Feedback'] == 'Positive').astype(int)

radar_df_new = data.groupby('Educational Qualifications').agg(
    Average_Age=('Age', 'mean'),
    Average_Family_Size=('Family size', 'mean'),
    Proportion_Positive_Feedback=('Positive_Feedback', 'mean'),
    Proportion_Ordering_Online=('Output_Numeric', 'mean')
).reset_index()

scaler = MinMaxScaler()
radar_df_normalized = pd.DataFrame(scaler.fit_transform(radar_df_new.iloc[:, 1:]), columns=radar_df_new.columns[1:])
radar_df_normalized['Educational Qualifications'] = radar_df_new['Educational Qualifications']

categories_new = list(radar_df_normalized)[1:]
N_new = len(categories_new)

angles_new = [n / float(N_new) * 2 * 3.14159265359 for n in range(N_new)]
angles_new += angles_new[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

plt.xticks(angles_new[:-1], categories_new)

ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
plt.ylim(0,1)

for i, row in radar_df_normalized.iterrows():
    data = radar_df_normalized.loc[i].drop('Educational Qualifications').tolist()
    data += data[:1]
    ax.plot(angles_new, data, linewidth=2, linestyle='solid', label=radar_df_normalized['Educational Qualifications'][i])
    ax.fill(angles_new, data, alpha=0.1)

plt.title('Enhanced Radar Chart for Educational Qualifications', size=20, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()
```


    
![png](output_29_0.png)
    



```python
data = pd.read_csv('/kaggle/input/online-food-dataset/onlinefoods.csv')
```

**The Bar Plot below shows number of Positive and Negative Feedbacks**


```python
sentiment_counts = data['Feedback'].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color='skyblue')
plt.title('Feedback Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

sentiment_counts

```


    
![png](output_32_0.png)
    





    Feedback
    Positive     317
    Negative      71
    Name: count, dtype: int64



**Random Forest Machine Learning (ML) Classification Model to predict feedback as Positive or Negative**


```python
encoder = LabelEncoder()
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Output', 'Unnamed: 12']
for feature in categorical_features:
    data[feature] = encoder.fit_transform(data[feature])

X = data.drop(['Feedback', 'latitude', 'longitude', 'Pin code'], axis=1)
y = encoder.fit_transform(data['Feedback'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
report_rf = classification_report(y_test, y_pred)

print(accuracy_rf)
print(report_rf)
```

    0.8974358974358975
                  precision    recall  f1-score   support
    
               0       0.62      0.73      0.67        11
               1       0.95      0.93      0.94        67
    
        accuracy                           0.90        78
       macro avg       0.78      0.83      0.80        78
    weighted avg       0.91      0.90      0.90        78
    
    

Random Forest model performs well in order to predict classification.
