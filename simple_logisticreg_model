import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.metrics import precision_recall_curve,plot_precision_recall_curve,plot_roc_curve

#data
df = pd.read_csv('hearing_test.csv')
df.head()

#getting information about the dataset
#info and describe
print(df.info())
print(df.describe())

#data visualizations 
print(df['test_result'].value_counts())

#visualizing the data using seaborn and matplotib.pyplot
sns.countplot(data=df,x='test_result')
sns.boxplot(x='test_result',y='age',data=df)
sns.boxplot(x='test_result',y='physical_score',data=df)
sns.scatterplot(x='age',y='physical_score',data=df,hue='test_result')
sns.pairplot(df,hue='test_result')
sns.heatmap(df.corr(),annot=True)
sns.scatterplot(x='physical_score',y='test_result',data=df)
sns.scatterplot(x='age',y='test_result',data=df)

#visualizing the data in 3 dimension
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['age'],df['physical_score'],df['test_result'],c=df['test_result'])

#diving the data to x and y
X = df.drop('test_result',axis=1)
y = df['test_result']

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#using logisticregression which is already impported
# help(LogisticRegression)
# help(LogisticRegressionCV)

#creating model
log_model = LogisticRegression()

#fitting 
log_model.fit(scaled_X_train,y_train)

#viewing the coefficients
print(log_model.coef_)

#making predictions on test data
y_pred = log_model.predict(scaled_X_test)
print(y_pred)

#calcuating accuracy,etc...
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
plot_confusion_matrix(log_model,scaled_X_test,y_test)

# Scaled so highest value=1
plot_confusion_matrix(log_model,scaled_X_test,y_test,normalize='true')
print(classification_report(y_test,y_pred))
print(X_train.iloc[0])

# 0% probability of 0 class
# 100% probability of 1 class
log_model.predict_proba(X_train.iloc[0].values.reshape(1, -1))
log_model.predict(X_train.iloc[0].values.reshape(1, -1))

plot_precision_recall_curve(log_model,scaled_X_test,y_test)
plot_roc_curve(log_model,scaled_X_test,y_test)
