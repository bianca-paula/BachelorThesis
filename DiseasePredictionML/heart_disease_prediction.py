# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
# from sklearn import preprocessing
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, plot_confusion_matrix
# from sklearn.naive_bayes import GaussianNB
# from kmodes.kmodes import KModes
#
# #read the dataset from the csv file
# dataset = pd.read_csv("D:/BachelorThesis/DiseasePredictionML/data/cardio_train.csv", sep=";")
#
# print(dataset.head())
#
# print(dataset.info())
#
# print(dataset.isnull().values.any())
#
# dataset['years'] = (dataset['age'] / 365).round().astype('int')
#
# #!!!!remove records where height and weight are below 2% or above 98% of a given range
# dataset.drop(dataset[(dataset['height'] > dataset['height'].quantile(0.975)) | (dataset['height'] < dataset['height'].quantile(0.024))].index,inplace=True)
# dataset.drop(dataset[(dataset['weight'] > dataset['weight'].quantile(0.975)) | (dataset['weight'] < dataset['weight'].quantile(0.024))].index,inplace=True)
#
# #remove records where the diastolic pressure is higher than systolic
# print("Diastolic pressure is higher than systolic in {0} cases".format(dataset[dataset['ap_lo'] > dataset['ap_hi']].shape[0]))
# print("Blood pressure is negative in {0} cases".format(dataset[dataset['ap_hi']/dataset['ap_lo'] <0].shape[0]))
#
# dataset.drop(dataset[(dataset['ap_hi'] > dataset['ap_hi'].quantile(0.98)) | (dataset['ap_hi'] < dataset['ap_hi'].quantile(0.019))].index,inplace=True)
# dataset.drop(dataset[(dataset['ap_lo'] > dataset['ap_lo'].quantile(0.98)) | (dataset['ap_lo'] < dataset['ap_lo'].quantile(0.019))].index,inplace=True)
#
# print("Diastolic pressure is higher than systolic in {0} cases".format(dataset[dataset['ap_lo'] > dataset['ap_hi']].shape[0]))
# print("Blood pressure is negative in {0} cases".format(dataset[dataset['ap_hi']/dataset['ap_lo'] <0].shape[0]))
#
#
#
# dataset.height = dataset['height']/100 #convert cm to m
# dataset.age = dataset.years
# dataset=dataset.drop(['years'], axis=1)
# dataset=dataset.drop(['id'], axis=1)
#
# # age_bin in quinquenium 5 years spam
# dataset['age_bin'] = pd.cut(dataset['age'], [0,20,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
#                               labels=['0-20', '20-30', '30-35', '35-40','40-45','45-50','50-55','55-60','60-65','65-70','70-75','75-80','80-85','85-90','90-95','95-100'])
#
# # Adding Body Mass Index
# dataset['bmi'] = dataset['weight']/((dataset['height']/100)**2)
# # transforming the column bmi in Body Mass Index Classes (1 to 6)
# rating = []
# for row in dataset['bmi']:
#     if row < 18.5:
#         rating.append(1)  # UnderWeight
#     elif row > 18.5 and row < 24.9:
#         rating.append(2)  # NormalWeight
#     elif row > 24.9 and row < 29.9:
#         rating.append(3)  # OverWeight
#     elif row > 29.9 and row < 34.9:
#         rating.append(4)  # ClassObesity_1
#     elif row > 34.9 and row < 39.9:
#         rating.append(5)  # ClassObesity_2
#     elif row > 39.9 and row < 49.9:
#         rating.append(6)  # ClassObesity_3
#     elif row > 49.9:
#         rating.append('Error')
#
#     else:
#         rating.append('Not_Rated')
#
# # inserting Column
# dataset['BMI_Class'] = rating
#
# # general distribuition
# dataset["BMI_Class"].value_counts(normalize=True)
#
# # creating a Column for MAP
# dataset['MAP'] = ((2* dataset['ap_lo']) + dataset['ap_hi']) / 3
#
# # Creating Classes for MAP
# map_values = []
# for row in dataset['MAP']:
#     if row < 69.9:
#         map_values.append(1)  # Low
#     elif row > 70 and row < 79.9:
#         map_values.append(2)  # Normal
#     elif row > 79.9 and row < 89.9:
#         map_values.append(3)  # Normal
#     elif row > 89.9 and row < 99.9:
#         map_values.append(4)  # Normal
#     elif row > 99.9 and row < 109.9:
#         map_values.append(5)  # High
#     elif row > 109.9 and row < 119.9:
#         map_values.append(6)  # Normal
#     elif row > 119.9:
#         map_values.append(7)
#
#     else:
#         map_values.append('Not_Rated')
#
# #inserting MAP_Class Column
# dataset['MAP_Class'] = map_values
# # Reordering Columns
# dataset= dataset[["gender","height","weight","bmi","ap_hi","ap_lo","MAP","age","age_bin","BMI_Class","MAP_Class","cholesterol","gluc","smoke","active","cardio"]]
# dataset.head()
#
# # only Categorical Data Columns DataFrame
# dataset_cat = dataset[["gender","age_bin","BMI_Class","MAP_Class","cholesterol","gluc","smoke","active","cardio",]]
# le = preprocessing.LabelEncoder()
# dataset_cat = dataset_cat.apply(le.fit_transform)
# dataset_cat.head()
#
# dataset_male = dataset_cat.query("gender == 0")
# dataset_female = dataset_cat.query("gender == 1")
#
#
# #2 clusters
#
# # female data
# # Building the model with using K-Mode with "Huang" initialization
# km_huang = KModes(n_clusters=2, init = "Huang", n_init = 5, verbose=0)
# clusters_huang_1 = km_huang.fit_predict(dataset_female)
# clusters_huang_1
#
# # male data
# # Building the model with using K-Mode with "Huang" initialization
# km_huang = KModes(n_clusters=2, init = "Huang", n_init = 5, verbose=0)
# clusters_huang_2 = km_huang.fit_predict(dataset_male)
# clusters_huang_2
#
#
# dataset_female.insert(0,"Cluster", clusters_huang_1, True)
# dataset_male.insert(0, "Cluster", clusters_huang_2, True)
#
# dataset_male["Cluster"].replace({0:2, 1:3}, inplace=True)
# # merging female and male data
# df_clusters = pd.concat([dataset_female, dataset_male], ignore_index=True, sort=False)
#
#
# corr = df_clusters.corr()
# cmap = sns.diverging_palette(2, 15, as_cmap=True)
# # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(20, 10))
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.7, center=0,annot = True,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
#
# # separating clusters
# df_female_c0 = dataset_female[dataset_female["Cluster"] == 0]
# df_female_c1 = dataset_female[dataset_female["Cluster"] == 1]
#
#
# # separating clusters
# df_male_c2 = dataset_male[dataset_male["Cluster"] == 2]
# df_male_c3 = dataset_male[dataset_male["Cluster"] == 3]
#
# #list of columns
# my_list = df_clusters.columns.values.tolist()
# my_list
#
# # function to calc % in column
# def function(dataframe,valor):
#     my_list = dataframe.columns.values.tolist()
#     for x in my_list[2:]:
#         for y in range(valor):
#             percentages = round(((dataframe[x]== y) & (dataframe['cardio']== 1)).sum()/ (dataframe[x]== y).sum()* 100, 2)
#             print(percentages)
# function(df_male_c3,8)
#
# # dictionary with list object in values
# # placed by hand...
# details = {
#     'age_bin_0': [0.0, 0.0, 0.0, 0.0],
#     'age_bin_1': [7.62, 92.44, 8.48, 100.0],
#     'age_bin_2': [12.94, 92.46, 13.73, 100.0],
#     'age_bin_3': [19.73, 92.21, 16.17, 100.0],
#     'age_bin_4': [36.29, 100.0, 34.82, 100.0],
#     'age_bin_5': [16.6, 84.63, 0.0, 76.59],
#     'age_bin_6': [38.56, 93.29, 38.23, 100.0],
#
#     'BMI_Class_0': [12.07, 100.0, 9.76, 100.0],
#     'BMI_Class_1': [31.1, 100.0, 30.77, 100.0],
#     'BMI_Class_2': [8.8, 86.34, 0.0, 78.11],
#     'BMI_Class_3': [32.73, 94.86, 22.18, 100.0],
#     'BMI_Class_4': [37.62, 94.29, 27.91, 100.0],
#     'BMI_Class_5': [66.67, 100.0, 31.54, 100.0],
#
#     'MAP_Class_0': [12.55, 85.71, 10.2, 70.29],
#     'MAP_Class_1': [11.94, 75.34, 10.1, 75.49],
#     'MAP_Class_2': [25.28, 100.0, 17.03, 82.55],
#     'MAP_Class_3': [19.89, 85.49, 46.41, 94.35],
#     'MAP_Class_4': [60.13, 97.42, 59.48, 97.17],
#     'MAP_Class_5': [61.4, 95.04, 58.67, 97.95],
#
#     'cholesterol_0': [22.06, 89.16, 18.84, 85.04],
#     'cholesterol_1': [29.81, 92.4, 26.92, 90.33],
#     'cholesterol_2': [48.41, 92.0, 44.38, 95.2],
#
#     'gluc_0': [23.69, 90.18, 20.78, 86.95],
#     'gluc_1': [28.45, 90.39, 24.72, 90.59],
#     'gluc_2': [31.87, 89.4, 31.12, 91.51],
#
#     'smoke_0': [25.93, 90.48, 21.73, 87.7],
#     'smoke_1': [18.9, 88.86, 19.49, 88.43],
#
#     'active_0': [29.93, 91.7, 24.19, 88.99],
#     'active_1': [23.16, 89.73, 21.1, 87.38],
# }
#
# # creating a Dataframe object from dictionary
# # with custom indexing
# df_pc = pd.DataFrame(details, index = ['Cluster 0','Cluster 1','Cluster 2','Cluster 3'])
# df_pc.head()
#
# # creating separate dfs
# df_agebins = df_pc.loc[:,"age_bin_0":"age_bin_6"]
# df_bmi = df_pc.loc[:,"BMI_Class_0":"BMI_Class_5"]
# df_map = df_pc.loc[:,"MAP_Class_0":"MAP_Class_5"]
# df_chol = df_pc.loc[:,"cholesterol_0":"cholesterol_2"]
# df_gluc = df_pc.loc[:,"gluc_0":"gluc_2"]
# df_smokers = df_pc.loc[:,"smoke_0":"smoke_1"]
# df_actives = df_pc.loc[:,"active_0":"active_1"]
#
# le = preprocessing.LabelEncoder()
# df_ml = df_clusters.apply(le.fit_transform)
# df_ml.describe()
# target_name = 'cardio'
# data_target = df_clusters[target_name]
# data = df_clusters.drop([target_name], axis=1)
# #separate into 30/70%
# train, test, target, target_test = train_test_split(data, data_target, test_size=0.3, random_state=0)
# #%% split training set to validation set
# Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.3, random_state=0)
#
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(train, target)
# acc_decision_tree = round(decision_tree.score(train, target) * 100, 2)
# acc_decision_tree
#
# acc_test_decision_tree = round(decision_tree.score(test, target_test) * 100, 2)
# acc_test_decision_tree
#
# y_pred_dt = decision_tree.predict(Xval)
#
# # Random Forest
#
# random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 300]}, cv=5).fit(train, target)
# random_forest.fit(train, target)
# acc_random_forest = round(random_forest.score(train, target) * 100, 2)
# print("FOREST")
# print(acc_random_forest,random_forest.best_params_)
#
# acc_test_random_forest = round(random_forest.score(test, target_test) * 100, 2)
# print(acc_test_random_forest)
#
# y_pred_df = random_forest.predict(Xval)
#
#
#
# # X= dataset.drop('cardio', axis=1)
# # y = dataset.cardio
# # print("X is: ")
# # print(X)
# # print("y is: ")
# # print(y)
#
# # #if we do not set train_size and test_size, they will be by default 25% and 75%
# # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=24)
# #
# #
# # # Defining scoring metric for k-fold cross validation
# # def cv_scoring(estimator, X, y):
# #     return accuracy_score(y, estimator.predict(X))
# #
# #
# # # Initializing Models
# # models = {
# #     "XGBClassifier": XGBClassifier(),
# #     "Gradient Boosting": GradientBoostingClassifier(),
# #     "Random Forest": RandomForestClassifier(random_state=18)
# # }
# #
# # # Producing cross validation score for the models
# # for model_name in models:
# #     model = models[model_name]
# #     scores = cross_val_score(model, X, y, cv=10,
# #                              n_jobs=-1,
# #                              scoring=cv_scoring)
# #     print("==" * 30)
# #     print(model_name)
# #     print(f"Scores: {scores}")
# #     print(f"Mean Score: {np.mean(scores)}")
# #
# #
# #
# # # The XGBoost stands for eXtreme Gradient Boosting, which is a boosting algorithm based on gradient boosted decision
# # # trees algorithm. XGBoost applies a better regularization technique to reduce overfitting, and it is one of the
# # # differences from the gradient boosting.
# #
# # # xgb = XGBClassifier(learning_rate = 0.00292, n_estimators = 2900, max_depth=7, random_state=100)
# # # xgb= XGBClassifier()
# # # # xgb.load_model('model.json')
# # # xgb.fit(X_train, y_train)
# # # y_pred = xgb.predict(X_train)
# # # predictions = [round(value) for value in y_pred]
# # # accuracy=accuracy_score(y_train, predictions)
# # # print("Accuracy on train by XGBC: %.4f%%" % (accuracy * 100.0))
# #
# # xgb_model = XGBClassifier()
# # xgb_model.load_model('model_xgb.json')
# #
# # xgb_model.fit(X_train, y_train)
# # preds = xgb_model.predict(X_test)
# # print(f"Accuracy on train data by xgb_model\
# # : {accuracy_score(y_train, xgb_model.predict(X_train))*100}")
# #
# # print(f"Accuracy on test data by xgb_model\
# # : {accuracy_score(y_test, preds)*100}")
# #
# # gb_model = GradientBoostingClassifier()
# # # xgb_model.load_model('model_gb.json')
# # gb_model.fit(X_train, y_train)
# # preds = gb_model.predict(X_test)
# # print(f"Accuracy on train data by gb_model\
# # : {accuracy_score(y_train, gb_model.predict(X_train))*100}")
# #
# # print(f"Accuracy on test data by gb\
# # : {accuracy_score(y_test, preds)*100}")
# #
# #
# # rf_model = RandomForestClassifier(n_estimators=100, random_state=18)
# # # rf_model.load_model('model_rf.json')
# # rf_model.fit(X_train, y_train)
# # preds = rf_model.predict(X_test)
# # print(f"Accuracy on train data by Random Forest Classifier\
# # : {accuracy_score(y_train, rf_model.predict(X_train)) * 100}")
# #
# # print(f"Accuracy on test data by Random Forest Classifier\
# # : {accuracy_score(y_test, preds) * 100}")
# #
# #
# #
# # xgb_model.save_model('model_xgb.json')
#
#
#
