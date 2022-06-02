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
from kmodes.kmodes import KModes
# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# import pandas_profiling as pp
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from matplotlib import rcParams
import pandas_profiling as pp
# Load the dataset from the csv file
dataset_all = pd.read_csv("./data/cardio_train.csv", sep=";")
# Show general information about data
dataset_all.info()
# See the first 5 rows of the data
print(dataset_all.head())
print("--------------Check for duplicates---------------")
print(dataset_all.duplicated().any())

#check how many samples are for each label
dataset_all['cardio'].value_counts()

#removing Outliers
# Let's remove weights and heights, that fall below 2.5% or above 97.5% of a given range.
dataset_all.drop(dataset_all[(dataset_all['height'] > dataset_all['height'].quantile(0.975)) | (dataset_all['height'] < dataset_all['height'].quantile(0.025))].index,inplace=True)
dataset_all.drop(dataset_all[(dataset_all['weight'] > dataset_all['weight'].quantile(0.975)) | (dataset_all['weight'] < dataset_all['weight'].quantile(0.025))].index,inplace=True)

dataset_all.drop(dataset_all[(dataset_all['ap_hi'] > dataset_all['ap_hi'].quantile(0.975)) | (dataset_all['ap_hi'] < dataset_all['ap_hi'].quantile(0.025))].index,inplace=True)
dataset_all.drop(dataset_all[(dataset_all['ap_lo'] > dataset_all['ap_lo'].quantile(0.975)) | (dataset_all['ap_lo'] < dataset_all['ap_lo'].quantile(0.025))].index,inplace=True)
print("Diastolic pressure is higher than systolic one in {0} cases".format(dataset_all[dataset_all['ap_lo']> dataset_all['ap_hi']].shape[0]))

# transform the column age, that is measured in days, in years
dataset_all['years'] = (dataset_all['age'] / 365).round().astype('int')
dataset_all.drop(['age'], axis='columns', inplace=True)
#drop the id column
dataset_all.drop(['id'], axis='columns', inplace=True)
# rcParams['figure.figsize'] = 11, 8
# sns.countplot(x='years', hue='cardio', data = dataset_all, palette="Set2");
# plt.savefig("./exploratory-data-analysis-plots/age-predisposition.jpg")
# # plt.show()
#
# df_long = pd.melt(dataset_all, id_vars=['cardio'], value_vars=['cholesterol','gluc', 'smoke', 'alco', 'active'])
# sns.catplot(x="variable", hue="value", col="cardio",
#                 data=df_long, kind="count");
# plt.savefig("./exploratory-data-analysis-plots/categorical-variables-distribution.jpg")
#
# corr = dataset_all.corr()
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot = True,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5});
# plt.savefig("./exploratory-data-analysis-plots/heatmap.jpg")

# age_bin in quinquenium 5 years spam
dataset_all['age_bin'] = pd.cut(dataset_all['years'], [0,20,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
                              labels=['0-20', '20-30', '30-35', '35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'])
print(dataset_all['age_bin'])
# Adding Body Mass Index
dataset_all['bmi'] = round(dataset_all['weight']/((dataset_all['height']/100)**2),2)
dataset_all['bmi']

# transforming the column bmi in Body Mass Index Classes (1 to 6)
rating = []
for row in dataset_all['bmi']:
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
dataset_all['BMI_Class'] = rating
dataset_all['BMI_Class']
dataset_all.drop(dataset_all[dataset_all['BMI_Class']=='Error'].index,inplace=True)
dataset_all.drop(dataset_all[dataset_all['BMI_Class']=='Not_Rated'].index,inplace=True)


# only Categorical Data Columns DataFrame
dataset_categorical = dataset_all[["gender","age_bin","BMI_Class","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active","cardio"]]

le = preprocessing.LabelEncoder()
dataset_categorical['age_bin'] = le.fit_transform(dataset_categorical['age_bin'])
dataset_categorical['BMI_Class'] = le.fit_transform(dataset_categorical['BMI_Class'])
dataset_categorical
print("Here1")
dataset_male = dataset_categorical.query("gender == 2")
dataset_female = dataset_categorical.query("gender == 1")

# female data
# Building the model with using K-Mode with "Huang" initialization
km_huang = KModes(n_clusters=4, init = "Huang", n_init = 5, verbose=0)
clusters_huang_1 = km_huang.fit_predict(dataset_female)
clusters_huang_1
print("Here2")
# male data
# Building the model with using K-Mode with "Huang" initialization
km_huang = KModes(n_clusters=4, init = "Huang", n_init = 5, verbose=0)
clusters_huang_2 = km_huang.fit_predict(dataset_male)
clusters_huang_2
print("Here3")
dataset_female.insert(0,"Cluster", clusters_huang_1, True)
dataset_male.insert(0, "Cluster", clusters_huang_2, True)
# replacing cluster column values to merge dataframes after
dataset_male["Cluster"].replace({0:4, 1:5, 2:6, 3:7}, inplace=True)

dataset_clusters = pd.concat([dataset_female, dataset_male], ignore_index=True, sort=False)
dataset_clusters.to_csv(path_or_buf='./data/cardiovascular_disease_prediction_with_clusters2.csv', sep=';')
target_name = 'cardio'
data_target = dataset_clusters[target_name]
data = dataset_clusters.drop([target_name], axis=1)

train, test, target, target_test = train_test_split(data, data_target, test_size=0.4, random_state=0)

# Random Forest
print("Here4")
random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100,300]}, cv=10)
random_forest.fit(train, target)
acc_random_forest = round(random_forest.score(train, target) * 100, 2)
print(acc_random_forest,random_forest.best_params_)