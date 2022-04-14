import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
# !pip install kmodes
from kmodes.kmodes import KModes
# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

#read the dataset from the csv file
all_dataset = pd.read_csv("./data/cardio_train.csv", sep=";")

#check general information about data
print(all_dataset.head())

#removing Outliers
# Let's remove weights and heights, that fall below 2.5% or above 97.5% of a given range.
all_dataset.drop(all_dataset[(all_dataset['height'] > all_dataset['height'].quantile(0.975)) | (all_dataset['height'] < all_dataset['height'].quantile(0.025))].index,inplace=True)
all_dataset.drop(all_dataset[(all_dataset['weight'] > all_dataset['weight'].quantile(0.975)) | (all_dataset['weight'] < all_dataset['weight'].quantile(0.025))].index,inplace=True)


#In addition, in some cases diastolic pressure is higher than systolic, which is also  incorrect. How many records are inaccurate in terms of blood pressure?
# shape[0] - take number of rows
print("Diastolic pressure is higher than systolic one in {0} cases".format(all_dataset[all_dataset['ap_lo']> all_dataset['ap_hi']].shape[0]))
all_dataset.drop(all_dataset[(all_dataset['ap_hi'] > all_dataset['ap_hi'].quantile(0.975)) | (all_dataset['ap_hi'] < all_dataset['ap_hi'].quantile(0.025))].index,inplace=True)
all_dataset.drop(all_dataset[(all_dataset['ap_lo'] > all_dataset['ap_lo'].quantile(0.975)) | (all_dataset['ap_lo'] < all_dataset['ap_lo'].quantile(0.025))].index,inplace=True)
print("Diastolic pressure is higher than systolic one in {0} cases".format(all_dataset[all_dataset['ap_lo']> all_dataset['ap_hi']].shape[0]))

# transforming the column AGE(measured in days) for Years
all_dataset['years'] = (all_dataset['age'] / 365).round().astype('int')
all_dataset.drop(['age'], axis='columns', inplace=True)
all_dataset.drop(['id'], axis='columns', inplace=True)
print(all_dataset.describe())

# age_bin in quinquenium 5 years spam
all_dataset['age_bin'] = pd.cut(all_dataset['years'], [0,20,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
                              labels=['0-20', '20-30', '30-35', '35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'])
print(all_dataset['age_bin'])

# Adding Body Mass Index
all_dataset['bmi'] = all_dataset['weight']/((all_dataset['height']/100)**2)
# transforming the column bmi in Body Mass Index Classes (1 to 6)
rating = []
for row in all_dataset['bmi']:
    if row < 18.5:
        rating.append(1)  # UnderWeight
    elif row > 18.5 and row < 24.9:
        rating.append(2)  # NormalWeight
    elif row > 24.9 and row < 29.9:
        rating.append(3)  # OverWeight
    elif row > 29.9 and row < 34.9:
        rating.append(4)  # ClassObesity_1
    elif row > 34.9 and row < 39.9:
        rating.append(5)  # ClassObesity_2
    elif row > 39.9 and row < 49.9:
        rating.append(6)  # ClassObesity_3
    elif row > 49.9:
        rating.append('Error')

    else:
        rating.append('Not_Rated')
# inserting Column
all_dataset['BMI_Class'] = rating
all_dataset['BMI_Class']

# general distribuition
all_dataset["BMI_Class"].value_counts(normalize=True)

# creating a Column for MAP
all_dataset['MAP'] = ((2* all_dataset['ap_lo']) + all_dataset['ap_hi']) / 3
# Creating Classes for MAP
map_values = []
for row in all_dataset['MAP']:
    if row < 69.9:
        map_values.append(1)  # Low
    elif row > 70 and row < 79.9:
        map_values.append(2)  # Normal
    elif row > 79.9 and row < 89.9:
        map_values.append(3)  # Normal
    elif row > 89.9 and row < 99.9:
        map_values.append(4)  # Normal
    elif row > 99.9 and row < 109.9:
        map_values.append(5)  # High
    elif row > 109.9 and row < 119.9:
        map_values.append(6)  # Normal
    elif row > 119.9:
        map_values.append(7)

    else:
        map_values.append('Not_Rated')

# inserting MAP_Class Column
all_dataset['MAP_Class'] = map_values

# Reordering Columns
all_dataset= all_dataset[["gender","height","weight","bmi","ap_hi","ap_lo","MAP","years","age_bin","BMI_Class","MAP_Class","cholesterol","gluc","smoke","active","cardio"]]
all_dataset.head()

# only Categorical Data Columns DataFrame
df_cat = all_dataset[["gender","age_bin","BMI_Class","MAP_Class","cholesterol","gluc","smoke","active","cardio"]]
df_cat

le = preprocessing.LabelEncoder()
df_cat = df_cat.apply(le.fit_transform)
df_male = df_cat.query("gender == 0")
df_female = df_cat.query("gender == 1")

# # Elbow curve to find optimal K in Huang init
# cost = []
# K = range(1,10)
# for num_clusters in list(K):
#     kmode = KModes(n_clusters=num_clusters, init = "Huang", n_init = 5, verbose=0)
#     kmode.fit_predict(df_male)
#     cost.append(kmode.cost_)

# plt.plot(K, cost, 'bx-')
# plt.xlabel('No. of clusters')
# plt.ylabel('Cost')
# plt.title('Elbow Method For Optimal k')
# plt.show()

# female data
# Building the model with using K-Mode with "Huang" initialization
km_huang = KModes(n_clusters=2, init = "Huang", n_init = 5, verbose=0)
clusters_huang_1 = km_huang.fit_predict(df_female)
clusters_huang_1

# male data
# Building the model with using K-Mode with "Huang" initialization
km_huang = KModes(n_clusters=2, init = "Huang", n_init = 5, verbose=0)
clusters_huang_2 = km_huang.fit_predict(df_male)
clusters_huang_2

df_female.insert(0,"Cluster", clusters_huang_1, True)
df_male.insert(0, "Cluster", clusters_huang_2, True)
# replacing cluster column values to merge dataframes after
df_male["Cluster"].replace({0:2, 1:3}, inplace=True)
# merging female and male data
df_clusters = pd.concat([df_female, df_male], ignore_index=True, sort=False)
corr = df_clusters.corr()
cmap = sns.diverging_palette(2, 15, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 10))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.7, center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# separating clusters
df_female_c0 = df_female[df_female["Cluster"] == 0]
df_female_c1 = df_female[df_female["Cluster"] == 1]

# separating clusters
df_male_c2 = df_male[df_male["Cluster"] == 2]
df_male_c3 = df_male[df_male["Cluster"] == 3]

#list of columns
my_list = df_clusters.columns.values.tolist()

# function to calc % in column
def function(dataframe,valor):
    my_list = dataframe.columns.values.tolist()
    for x in my_list[2:]:
        for y in range(valor):
            print("****************************************************************************************************")
            print("X= ")
            print(x)
            print("Data frame x= ")
            print(dataframe[x])
            print("Y= ")
            print(y)
            print("Cardio DF= ")
            print(dataframe['cardio'])
            percentages = round(((dataframe[x]== y) & (dataframe['cardio']== 1)).sum()/ (dataframe[x]== y).sum()* 100, 2)
            print(percentages)
            print("****************************************************************************************************")


# dictionary with list object in values
# placed by hand...
details = {
    'age_bin_0': [0.0, 0.0, 0.0, 0.0],
    'age_bin_1': [7.62, 92.44, 8.48, 100.0],
    'age_bin_2': [12.94, 92.46, 13.73, 100.0],
    'age_bin_3': [19.73, 92.21, 16.17, 100.0],
    'age_bin_4': [36.29, 100.0, 34.82, 100.0],
    'age_bin_5': [16.6, 84.63, 0.0, 76.59],
    'age_bin_6': [38.56, 93.29, 38.23, 100.0],

    'BMI_Class_0': [12.07, 100.0, 9.76, 100.0],
    'BMI_Class_1': [31.1, 100.0, 30.77, 100.0],
    'BMI_Class_2': [8.8, 86.34, 0.0, 78.11],
    'BMI_Class_3': [32.73, 94.86, 22.18, 100.0],
    'BMI_Class_4': [37.62, 94.29, 27.91, 100.0],
    'BMI_Class_5': [66.67, 100.0, 31.54, 100.0],

    'MAP_Class_0': [12.55, 85.71, 10.2, 70.29],
    'MAP_Class_1': [11.94, 75.34, 10.1, 75.49],
    'MAP_Class_2': [25.28, 100.0, 17.03, 82.55],
    'MAP_Class_3': [19.89, 85.49, 46.41, 94.35],
    'MAP_Class_4': [60.13, 97.42, 59.48, 97.17],
    'MAP_Class_5': [61.4, 95.04, 58.67, 97.95],

    'cholesterol_0': [22.06, 89.16, 18.84, 85.04],
    'cholesterol_1': [29.81, 92.4, 26.92, 90.33],
    'cholesterol_2': [48.41, 92.0, 44.38, 95.2],

    'gluc_0': [23.69, 90.18, 20.78, 86.95],
    'gluc_1': [28.45, 90.39, 24.72, 90.59],
    'gluc_2': [31.87, 89.4, 31.12, 91.51],

    'smoke_0': [25.93, 90.48, 21.73, 87.7],
    'smoke_1': [18.9, 88.86, 19.49, 88.43],

    'active_0': [29.93, 91.7, 24.19, 88.99],
    'active_1': [23.16, 89.73, 21.1, 87.38],
}

# creating a Dataframe object from dictionary
# with custom indexing
df_pc = pd.DataFrame(details, index = ['Cluster 0','Cluster 1','Cluster 2','Cluster 3'])
df_pc.head()

# creating separate dfs
df_agebins = df_pc.loc[:,"age_bin_0":"age_bin_6"]
df_bmi = df_pc.loc[:,"BMI_Class_0":"BMI_Class_5"]
df_map = df_pc.loc[:,"MAP_Class_0":"MAP_Class_5"]
df_chol = df_pc.loc[:,"cholesterol_0":"cholesterol_2"]
df_gluc = df_pc.loc[:,"gluc_0":"gluc_2"]
df_smokers = df_pc.loc[:,"smoke_0":"smoke_1"]
df_actives = df_pc.loc[:,"active_0":"active_1"]

le = preprocessing.LabelEncoder()
df_ml = df_clusters.apply(le.fit_transform)
df_ml.describe()


target_name = 'cardio'
data_target = df_clusters[target_name]
data = df_clusters.drop([target_name], axis=1)

train, test, target, target_test = train_test_split(data, data_target, test_size=0.4, random_state=0)
Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.4, random_state=0)
# Random Forest

random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [300]}, cv=5)
random_forest.fit(train, target)
acc_random_forest = round(random_forest.score(train, target) * 100, 2)
print(acc_random_forest,random_forest.best_params_)
pickle.dump(random_forest, open('./model.sav', 'wb'))
# input_data = [{'Cluster':2, 'gender':0,	'age_bin':4,	'BMI_Class':1,	'MAP_Class':2,	'cholesterol':2,	'gluc':0,	'smoke':0,	'active':0}]
# #2---	0	4	1	2	2	0	0	0	1
# #3	0	4	4	3	2	0	0	1	1
# #data=[1,	4,	1,	2,	0,	0,	0,	1]
# #print(data)
# df_test = pd.DataFrame(input_data)
# #print(km_huang.predict(df_test))
# # # input_data2 = [{'Cluster':1, 'gender':1,	'age_bin':4,	'BMI_Class':1,	'MAP_Class':2,	'cholesterol':0,	'gluc':0,	'smoke':0,	'active':1}]
# # # df_test2 = pd.DataFrame(input_data2)
#
# pred = random_forest.predict(df_test)
# if pred == 0:
#     print("Congratulations, You are out of risk of getting any cardiovascular disease.")
# else:
#     print("You are at risk of getting a cardiovascular disease. Please take care of yourself and just do regular check up of yours.")
#
# acc_test_random_forest = round(random_forest.score(test, target_test) * 100, 2)
# acc_test_random_forest
#
# y_pred_df = random_forest.predict(Xval)
# print(y_pred_df)