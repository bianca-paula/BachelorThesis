import pickle

import pandas as pd

loaded_model = pickle.load(open('./model.sav', 'rb'))
input_data = [{'Cluster':2, 'gender':0,	'age_bin':4,	'BMI_Class':1,	'MAP_Class':2,	'cholesterol':2,	'gluc':0,	'smoke':0,	'active':0}]
df_test = pd.DataFrame(input_data)
pred = loaded_model.predict(df_test)
if pred == 0:
    print("Congratulations, You are out of risk of getting any cardiovascular disease.")
else:
    print("You are at risk of getting a cardiovascular disease. Please take care of yourself and just do regular check up of yours.")