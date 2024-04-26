# Check the bank for a loan

In this project, we have written a program to check customers for granting loans

## module

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
```

## Usage

Reading data from csv file and limiting them

```python
data = pd.read_csv('loan.csv')
data.head()
data.describe()
data.describe()
data.drop(['Loan_ID'],axis=1,inplace=True)
data['Gender'].value_counts()
```

Creating a bar plot using the matplotlib and seaborn libraries that shows the distribution of the number of gender-related data in a dataframe

```python
plt.close()
y = data['Gender'].value_counts() 
plt.figure(figsize=(12,3))
sns.barplot(x=list(y.index),y=y)
```

Data preprocessing section to convert non-numerical data into numerical data

```python
label_encoder = preprocessing.LabelEncoder()

data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Married'] = label_encoder.fit_transform(data['Married'])
data['Education'] = label_encoder.fit_transform(data['Education'])
data['Self_Employed'] = label_encoder.fit_transform(data['Self_Employed'])
data['Property_Area'] = label_encoder.fit_transform(data['Property_Area'])
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])


#for col in list(obj[obj].index):
#   data[col] = label_encoder.fit_transform(data[col])
```

show

```python
data.head()
```

Creating a bar plot using the matplotlib and seaborn libraries that shows the distribution of the number of gender-related data in a dataframe

```python
plt.close()
y = data['Gender'].value_counts() 
plt.figure(figsize=(12,3))
sns.barplot(x=list(y.index),y=y)
```

Calculate the number of zero data in the dataset

```python
data.isnull().sum()
```

This code checks the data for missing values and replaces them with the average values of its columns. Then, it displays the number of remaining missing values for each column

```python
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

data.isna().sum()
```

This code prepares data for use in training machine learning models and then divides it into training and test sets.

```python
x = data.drop(['Loan_Status'], axis=1)
y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
```

This code creates and then trains (fit) a logistic regression model for prediction based on the training data.

```python
model = LogisticRegression()
model.max_iter = 10000
model.fit(X_train, Y_train)
```

This line of code generates logistic regression model predictions based on testing data.

```python
y_pred = model.predict(X_test)
```

This line of code is used to calculate the accuracy of the model based on predictions and actual labels from the test data, then displays the accuracy as a percentage.

```python
print(metrics.accuracy_score(Y_test,y_pred) * 100, '%')
```

## Result

This project was written by Majid Tajanjari and the Aiolearn team, and we need your support!❤️

# بانک را برای دریافت وام بررسی کنید

در این پروژه برنامه ای برای بررسی مشتریان برای اعطای وام نوشته ایم

## ماژول

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
```

## نحوه استفاده

خواندن داده ها از فایل csv و محدود کردن آنها

```python
data = pd.read_csv('loan.csv')
data.head()
data.describe()
data.describe()
data.drop(['Loan_ID'],axis=1,inplace=True)
data['Gender'].value_counts()
```

ایجاد یک نمودار نواری با استفاده از کتابخانه‌های matplotlib و seaborn که توزیع تعداد داده‌های مربوط به جنسیت را در یک دیتا فریم نشان می‌دهد.

```python
plt.close()
y = data['Gender'].value_counts() 
plt.figure(figsize=(12,3))
sns.barplot(x=list(y.index),y=y)
```

بخش پیش پردازش داده ها برای تبدیل داده های غیر عددی به داده های عددی

```python
label_encoder = preprocessing.LabelEncoder()

data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Married'] = label_encoder.fit_transform(data['Married'])
data['Education'] = label_encoder.fit_transform(data['Education'])
data['Self_Employed'] = label_encoder.fit_transform(data['Self_Employed'])
data['Property_Area'] = label_encoder.fit_transform(data['Property_Area'])
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])


#for col in list(obj[obj].index):
#   data[col] = label_encoder.fit_transform(data[col])
```

نمایش

```python
data.head()
```

ایجاد یک نمودار نواری با استفاده از کتابخانه‌های matplotlib و seaborn که توزیع تعداد داده‌های مربوط به جنسیت را در یک دیتا فریم نشان می‌دهد.

```python
plt.close()
y = data['Gender'].value_counts() 
plt.figure(figsize=(12,3))
sns.barplot(x=list(y.index),y=y)
```

تعداد داده های صفر در مجموعه داده را محاسبه کنید

```python
data.isnull().sum()
```

این کد داده ها را برای مقادیر از دست رفته بررسی می کند و آنها را با مقادیر میانگین ستون های خود جایگزین می کند. سپس، تعداد مقادیر گمشده باقیمانده را برای هر ستون نمایش می دهد

```python
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

data.isna().sum()
```

این کد داده ها را برای استفاده در آموزش مدل های یادگیری ماشین آماده می کند و سپس آنها را به مجموعه های آموزشی و آزمایشی تقسیم می کند.

```python
x = data.drop(['Loan_Status'], axis=1)
y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
```

این کد یک مدل رگرسیون لجستیک را برای پیش‌بینی بر اساس داده‌های آموزشی ایجاد و سپس آموزش می‌دهد.

```python
model = LogisticRegression()
model.max_iter = 10000
model.fit(X_train, Y_train)
```

این خط کد، پیش‌بینی‌های مدل رگرسیون لجستیک را بر اساس داده‌های آزمایشی ایجاد می‌کند.

```python
y_pred = model.predict(X_test)
```

این خط کد برای محاسبه دقت مدل بر اساس پیش‌بینی‌ها و برچسب‌های واقعی از داده‌های تست استفاده می‌شود، سپس دقت را به صورت درصد نمایش می‌دهد.

```python
print(metrics.accuracy_score(Y_test,y_pred) * 100, '%')
```

## نتیجه

این پروژه توسط مجید تجن جاری و تیم Aiolearn نوشته شده است و ما به حمایت شما نیازمندیم!❤️    