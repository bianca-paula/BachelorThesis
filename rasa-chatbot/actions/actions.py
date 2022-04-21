# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
#
import pickle
import numpy as np
import pandas as pd
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

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


#D:\BachelorThesis\DiseasePredictionML\models\model_voting_classifier.sav

# input_data = [{'gender':1,	'age_bin':5,	'BMI_Class':1,	'ap_hi':120,	'ap_lo':80,	'cholesterol':1, 'gluc':1,	'smoke':0,'alco':0,	'active':1, 'cardio':''}]
# km_huang=pickle.load(open('./models/clusters_huang444.sav', 'rb'))
# df_test= pd.DataFrame(input_data)
# print(input_data)
# print(df_test)
# cluster=km_huang.predict(df_test)
# print(cluster)
#
# df_test.drop('cardio', axis=1, inplace=True)
# df_test.insert(0, 'Cluster', cluster, True)
# print(df_test)
# pred = loaded_random_forest_model.predict(df_test)
# if pred == 0:
#     print("Congratulations, You are out of risk of getting any cardiovascular disease.")
# else:
#     print("You are at risk of getting a cardiovascular disease. Please take care of yourself and just do regular check up of yours.")

class ActionPredictHeartDiseaseRisk(Action):

    def name(self) -> Text:
        return "action_predict_heart_disease_risk"

    def categorize_age_bin(self, age):
        age=int(age)
        if(age < 35):
            return 0
        elif (age >= 35 and age <40):
            return 1
        elif (age >= 40 and age <45):
            return 2
        elif (age >= 45 and age <50):
            return 3
        elif (age >= 50 and age <55):
            return 4
        elif (age >= 55 and age <60):
            return 5
        elif (age >= 60):
            return 6
        else:
            raise Exception("Invalid age!")

    def calculate_bmi(self, height, weight):
        bmi = round(int(weight)/((int(height)/100)**2),2)
        bmi_class = -1
        if bmi < 18.5:
            bmi_class = 0
        elif bmi > 18.5 and bmi < 24.9:
            bmi_class = 1  # NormalWeight
        elif bmi > 24.9 and bmi < 29.9:
            bmi_class = 2  # OverWeight
        elif bmi > 29.9 and bmi < 34.9:
            bmi_class = 3  # ClassObesity_1
        elif bmi > 34.9 and bmi < 39.9:
            bmi_class = 4  # ClassObesity_2
        elif bmi > 39.9:
            bmi_class = 5  # ClassObesity_3
        return bmi_class

    def get_gender(self, gender):
        if(gender == 'female'):
            return 1
        else:
            return 2

    def get_cholesterol_level(self, cholesterol_level):
        if(cholesterol_level=='normal'):
            return 1
        elif (cholesterol_level=='above normal'):
            return 2
        else:
            return 3

    def get_glucose_level(self, glucose_level):
        if(glucose_level=='normal'):
            return 1
        elif (glucose_level=='above normal'):
            return 2
        else:
            return 3

    def get_is_smoking(self, is_smoking):
        if(is_smoking == 'True'):
            return 1
        else:
            return 0
    def get_drinks_alcohol(self, drinks_alcohol):
        if(drinks_alcohol == 'True'):
            return 1
        else:
            return 0
    def get_is_physically_active(self, is_physically_active):
        if(is_physically_active == 'True'):
            return 1
        else:
            return 0



    def predict_risk(self, patient_information):
        age = self.categorize_age_bin(patient_information["age"])
        gender = self.get_gender(patient_information["gender"])
        bmi = self.calculate_bmi(patient_information["height"],patient_information["weight"])
        systolic_blood_pressure = patient_information["systolic_blood_pressure"]
        diastolic_blood_pressure = patient_information["diastolic_blood_pressure"]
        glucose_level = self.get_glucose_level(patient_information["glucose_level"])
        cholesterol_level = self.get_cholesterol_level(patient_information["cholesterol_level"])
        drinks_alcohol = self.get_drinks_alcohol(patient_information["drinks_alcohol"])
        is_smoking = self.get_is_smoking(patient_information["is_smoking"])
        is_physically_active = self.get_is_physically_active(patient_information["is_physically_active"])

        model_input = [{'gender':gender,'age_bin':age,'BMI_Class':bmi,'ap_hi':systolic_blood_pressure,'ap_lo':diastolic_blood_pressure,
                        'cholesterol':cholesterol_level, 'gluc':glucose_level,'smoke':is_smoking,'alco':drinks_alcohol,'active':is_physically_active, 'cardio':''}]
        loaded_voting_model = pickle.load(open('D:/BachelorThesis/DiseasePredictionML/models/model_voting_classifier.sav', 'rb'))
        km_huang_female = pickle.load(open('D:/BachelorThesis/DiseasePredictionML/models/km_huang111.sav', 'rb'))
        km_huang_male = pickle.load(open('D:/BachelorThesis/DiseasePredictionML/models/km_huang222.sav', 'rb'))
        cluster=-1
        df_test= pd.DataFrame(model_input)
        if(gender == 1):
            cluster=km_huang_female.predict(df_test)
        else:
            cluster=km_huang_male.predict(df_test)+4
        df_test.drop('cardio', axis=1, inplace=True)
        df_test.insert(0, 'Cluster', cluster, True)
        pred = loaded_voting_model.predict(df_test)
        if pred == 0:
            return False
        return True

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        age = tracker.get_slot("age")
        gender = tracker.get_slot("gender")
        height = tracker.get_slot("height")
        weight = tracker.get_slot("weight")
        systolic_blood_pressure = tracker.get_slot("systolic_blood_pressure")
        diastolic_blood_pressure = tracker.get_slot("diastolic_blood_pressure")
        glucose_level = tracker.get_slot("glucose_level")
        cholesterol_level = tracker.get_slot("cholesterol_level")
        drinks_alcohol = tracker.get_slot("drinks_alcohol")
        is_smoking = tracker.get_slot("is_smoking")
        is_physically_active = tracker.get_slot("is_physically_active")
        patient_information={
            "age":age,
            "gender":gender,
            "height":height,
            "weight":weight,
            "systolic_blood_pressure":systolic_blood_pressure,
            "diastolic_blood_pressure":diastolic_blood_pressure,
            "glucose_level":glucose_level,
            "cholesterol_level":cholesterol_level,
            "drinks_alcohol":drinks_alcohol,
            "is_smoking":is_smoking,
            "is_physically_active":is_physically_active
        }
        prediction = self.predict_risk(patient_information)
        if prediction == True:
            dispatcher.utter_message("You are at risk of getting a cardiovascular disease. Please take care of yourself and just do regular check up of yours.")
        else:
            dispatcher.utter_message("Congratulations, you are out of risk of getting any cardiovascular disease.")

        return []


class ActionGivePersonalizedAdvice(Action):

    def name(self) -> Text:
        return "action_give_personalized_advice"

    # https: // www.nhs.uk / conditions / cardiovascular - disease /
    # https: // www.cdc.gov / heartdisease / risk_factors.htm
    def give_recommendation_user_is_smoking(self, is_smoking):
        if(is_smoking=='True'):
            advice="Smoking and other tobacco use is also a significant risk factor for CVD. \n" \
                   "The harmful substances in tobacco can damage and narrow your blood vessels, which increases your risk for heart conditions such as atherosclerosis and heart attack. \n" \
                   "Nicotine also raises blood pressure, while carbon monoxide from cigarette smoke reduces the amount of oxygen that your blood can carry. \n" \
                   "If you smoke, you should try to give up as soon as possible.\n" \
                   "Your GP can also provide you with advice and support. They can also prescribe medication to help you quit."
            return advice
        else:
            return "No advice for smoking"
    def give_recommendation_user_drinks_alcohol(self, drinks_alcohol):
        if(drinks_alcohol=='True'):
            advice="Drinking too much alcohol can raise blood pressure levels and the risk for heart disease. \n" \
                   "It also increases levels of triglycerides, a fatty substance in the blood which can increase the risk for heart disease. \n" \
                   "If you drink alcohol, try not to exceed the recommended limit of 14 alcohol units a week."
            return advice
        else:
            return ""
    def give_recommendation_user_is_physically_active(self):
        pass
    def give_recommendation_user_bmi(self):
        pass
    def give_recommendation_user_gender(self):
        pass
    def give_recommendation_user_age(self):
        pass
    def give_recommendation_user_systolic_blood_pressure(self):
        pass
    def give_recommendation_user_diastolic_blood_pressure(self):
        pass
    def give_recommendation_user_blood_pressure(self):
        pass
    def give_recommendation_user_glucose_level(self):
        pass
    def give_recommendation_user_cholesterol_level(self):
        pass
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_choice = tracker.get_slot("is_smoking")
        dispatcher.utter_message(self.give_recommendation_user_is_smoking(user_choice))

        return []


# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
