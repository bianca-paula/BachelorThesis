import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB

#read the dataset from the csv file
dataset = pd.read_csv("D:/BachelorThesis/DiseasePredictionML/data/cardio_train.csv", sep=";")

print(dataset.head())

print(dataset.info())

print(dataset.isnull().values.any())

dataset['years'] = (dataset['age'] / 365).round().astype('int')

#!!!!remove records where height and weight are below 2% or above 98% of a given range
dataset.drop(dataset[(dataset['height'] > dataset['height'].quantile(0.975)) | (dataset['height'] < dataset['height'].quantile(0.024))].index,inplace=True)
dataset.drop(dataset[(dataset['weight'] > dataset['weight'].quantile(0.975)) | (dataset['weight'] < dataset['weight'].quantile(0.024))].index,inplace=True)

#remove records where the diastolic pressure is higher than systolic
print("Diastolic pressure is higher than systolic in {0} cases".format(dataset[dataset['ap_lo'] > dataset['ap_hi']].shape[0]))
print("Blood pressure is negative in {0} cases".format(dataset[dataset['ap_hi']/dataset['ap_lo'] <0].shape[0]))

dataset.drop(dataset[(dataset['ap_hi'] > dataset['ap_hi'].quantile(0.98)) | (dataset['ap_hi'] < dataset['ap_hi'].quantile(0.019))].index,inplace=True)
dataset.drop(dataset[(dataset['ap_lo'] > dataset['ap_lo'].quantile(0.98)) | (dataset['ap_lo'] < dataset['ap_lo'].quantile(0.019))].index,inplace=True)

print("Diastolic pressure is higher than systolic in {0} cases".format(dataset[dataset['ap_lo'] > dataset['ap_hi']].shape[0]))
print("Blood pressure is negative in {0} cases".format(dataset[dataset['ap_hi']/dataset['ap_lo'] <0].shape[0]))



dataset.height = dataset['height']/100 #convert cm to m
dataset.age = dataset.years
dataset=dataset.drop(['years'], axis=1)
dataset=dataset.drop(['id'], axis=1)

X= dataset.drop('cardio', axis=1)
y = dataset.cardio
print("X is: ")
print(X)
print("y is: ")
print(y)

#if we do not set train_size and test_size, they will be by default 25% and 75%
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=32)

# The XGBoost stands for eXtreme Gradient Boosting, which is a boosting algorithm based on gradient boosted decision
# trees algorithm. XGBoost applies a better regularization technique to reduce overfitting, and it is one of the
# differences from the gradient boosting.

# xgb = XGBClassifier(learning_rate = 0.00292, n_estimators = 2900, max_depth=7, random_state=100)
gnb = GaussianNB()
xgb= XGBClassifier()
xgb.load_model('model.json')
xgb.fit(X_train, y_train, verbose=True)
gnb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
yy_pred = gnb.predict(X_test)
predictions = [round(value) for value in y_pred]
predictions2 = [round(value) for value in yy_pred]
accuracy=accuracy_score(y_test, predictions)
accuracy2=accuracy_score(y_test, predictions2)
print("Accuracy: %.4f%%" % (accuracy * 100.0))
print("Accuracy: %.4f%%" % (accuracy2 * 100.0))


xgb.save_model('model.json')


