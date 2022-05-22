# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
#
import re

import rasa_sdk
import wikipedia
from pprint import pprint
from haystack.document_stores import InMemoryDocumentStore, SQLDocumentStore
from haystack.nodes import FARMReader, TransformersReader, TfidfRetriever
from haystack.utils import convert_files_to_dicts,clean_wiki_text,fetch_archive_from_http, print_answers
from haystack.pipelines import ExtractiveQAPipeline
import logging
import tempfile
import requests
import pickle
import numpy as np
import pandas as pd
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import FollowupAction
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
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
from datetime import datetime
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores.faiss import FAISSDocumentStore

from compute_statistics.statistics import Statistics

def compute():
    statistics = Statistics(path="D:/BachelorThesis/rasa-chatbot/markers_values.csv")
    statistics.compute_statistics_gender_ratio()
    statistics.compute_statistics_cvd_prediction_ratio()
    statistics.compute_statistics_users_habits()
    statistics.compute_statistics_users_blood_pressure()
    statistics.compute_statistics_user_responses_by_month()
    statistics.compute_statistics_users_asked_faq()
    statistics.compute_statistics_users_asked_for_cvd_prediction()
    statistics.compute_statistics_users_found_advice_useful()

# class ActionComputeStatistics(Action):
#
#     def name(self) -> Text:
#         return "action_compute_statistics"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         statistics = Statistics(path="D:/BachelorThesis/rasa-chatbot/markers_values.csv")
#         statistics.compute_statistics_gender_ratio()
#         statistics.compute_statistics_cvd_prediction_ratio()
#         statistics.compute_statistics_users_habits()
#         statistics.compute_statistics_users_blood_pressure()
#         statistics.compute_statistics_user_responses_by_month()
#         statistics.compute_statistics_users_asked_faq()
#         statistics.compute_statistics_users_asked_for_cvd_prediction()
#         statistics.compute_statistics_users_found_advice_useful()
#
#
#         return []


class ActionSaveMarker(Action):

    def name(self) -> Text:
        return "action_save_marker"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any], marker: str) -> List[Dict[Text, Any]]:

        sender_id = tracker.current_state()['sender_id']
        session_idx = tracker.idx_after_latest_restart()
        add_marker_to_csv(sender_id,session_idx, marker)
        return []


save_marker = ActionSaveMarker()

def add_marker_to_csv(sender_id, session_idx, marker):
    dateTimeObj = datetime.now()
    data={
        'sender_id': [sender_id],
        'session_idx': [session_idx],
        'marker': [marker],
        'timestamp': [dateTimeObj]
    }
    # Make data frame of above data
    df = pd.DataFrame(data)

    # append data frame to CSV file
    df.to_csv('markers_values.csv', mode='a', index=False, header=False)
    compute_statistics(marker)

def get_wikipedia_articles(data_dir, topics=[]):
    for topic in topics:
        article = wikipedia.page(pageid=topic).content
        with open(f"{data_dir}/{topic}.txt", "w", encoding="utf-8") as f:
            f.write(article)

def get_bmi(height, weight):
    bmi = round(int(weight) / ((int(height) / 100) ** 2), 2)
    return bmi

def calculate_bmi(height, weight):
    bmi = get_bmi(height, weight)
    bmi_class = -1
    if bmi <= 18.5:
        bmi_class = 0
    elif bmi > 18.5 and bmi <= 24.9:
        bmi_class = 1  # NormalWeight
    elif bmi > 24.9 and bmi <= 29.9:
        bmi_class = 2  # OverWeight
    elif bmi > 29.9 and bmi <= 34.9:
        bmi_class = 3  # ClassObesity_1
    elif bmi > 34.9 and bmi <= 39.9:
        bmi_class = 4  # ClassObesity_2
    elif bmi > 39.9:
        bmi_class = 5  # ClassObesity_3
    return bmi_class

def categorize_age_bin(age):
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

def get_gender(gender):
    if(gender == 'female'):
        return 1
    else:
        return 2

def get_cholesterol_level(cholesterol_level):
    if(cholesterol_level.lower()=='normal'):
        return 1
    elif (cholesterol_level.lower()=='above normal'):
        return 2
    else:
        return 3

def get_glucose_level(glucose_level):
    if(glucose_level.lower()=='normal'):
        return 1
    elif (glucose_level.lower()=='above normal'):
        return 2
    else:
        return 3

def get_is_smoking(is_smoking):
    if(is_smoking == 'True'):
        return 1
    else:
        return 0

def get_drinks_alcohol(drinks_alcohol):
    if(drinks_alcohol == 'True'):
        return 1
    else:
        return 0

def get_is_physically_active(is_physically_active):
    if(is_physically_active == 'True'):
        return 1
    else:
        return 0

def get_blood_pressure_type(systolic, diastolic):
    if(systolic < 90 or diastolic < 60):
        return "low"
    elif(systolic < 120 and diastolic < 80):
        return "normal"
    elif((systolic>= 120 and systolic <= 129) and diastolic<80):
        return "elevated"
    else:
        return "hypertension"



class ActionSave_BeganConversation(Action):
    def name(self) -> Text:
        return "action_save_marker_user_began_conversation"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_user_began_conversation")
        return []

class ActionSave_FoundAdviceUseful(Action):
    def name(self) -> Text:
        return "action_save_marker_found_advice_useful"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_found_advice_useful")

        return []


def compute_statistics(marker):
    statistics = Statistics(path="D:/BachelorThesis/rasa-chatbot/markers_values.csv")
    if(marker=="marker_user_asked_faq"):
        statistics.compute_statistics_users_asked_faq()
    elif(marker=="marker_asked_for_advice" or marker=="marker_found_advice_useful"):
        statistics.compute_statistics_users_found_advice_useful()
    elif(marker=="marker_asked_for_cvd_prediction"):
        statistics.compute_statistics_users_asked_for_cvd_prediction()
        statistics.compute_statistics_user_responses_by_month()
    elif(marker=="marker_user_began_conversation"):
        statistics.compute_statistics_users_asked_for_cvd_prediction()
    elif(marker=="marker_cvd_prediction_user_female" or marker=="marker_cvd_prediction_user_male"):
        statistics.compute_statistics_gender_ratio()
    elif(marker=="marker_cvd_prediction_false" or marker=="marker_cvd_prediction_true" ):
        statistics.compute_statistics_cvd_prediction_ratio()
    elif(marker=="marker_cvd_prediction_user_drinks_alcohol" or marker=="marker_cvd_prediction_user_is_smoking" or marker=="marker_cvd_prediction_user_is_physically_active"):
        statistics.compute_statistics_users_habits()
    elif(marker=="marker_cvd_prediction_user_has_low_bp" or marker=="marker_cvd_prediction_user_has_normal_bp" or marker=="marker_cvd_prediction_user_has_elevated_bp"or marker=="marker_cvd_prediction_user_has_hypertension_bp"):
        statistics.compute_statistics_users_blood_pressure()

    else:
        pass
class ActionSave_AskedFAQ(Action):
    def name(self) -> Text:
        return "action_save_marker_user_asked_faq"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_user_asked_faq")
        return []




class ActionPredictHeartDiseaseRisk(Action):

    def name(self) -> Text:
        return "action_predict_heart_disease_risk"

    def predict_risk(self, patient_information):
        age = categorize_age_bin(patient_information["age"])
        gender = get_gender(patient_information["gender"])
        bmi = calculate_bmi(patient_information["height"],patient_information["weight"])
        systolic_blood_pressure = patient_information["systolic_blood_pressure"]
        diastolic_blood_pressure = patient_information["diastolic_blood_pressure"]
        glucose_level = get_glucose_level(patient_information["glucose_level"])
        cholesterol_level = get_cholesterol_level(patient_information["cholesterol_level"])
        drinks_alcohol = get_drinks_alcohol(patient_information["drinks_alcohol"])
        is_smoking = get_is_smoking(patient_information["is_smoking"])
        is_physically_active = get_is_physically_active(patient_information["is_physically_active"])

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
        sender_id = tracker.current_state()['sender_id']
        session_idx = tracker.idx_after_latest_restart()
        # add_marker_to_csv(sender_id,session_idx, 'marker_asked_for_cvd_prediction')
        save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_asked_for_cvd_prediction")
        # statistics.compute_statistics_users_asked_for_cvd_prediction()
        age = tracker.get_slot("age")
        gender = tracker.get_slot("gender")
        if (gender == 'female'):
            # add_marker_to_csv(sender_id,session_idx, 'marker_cvd_prediction_user_female')
            save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_cvd_prediction_user_female")
        else:
            # add_marker_to_csv(sender_id,session_idx, 'marker_cvd_prediction_user_male')
            save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_cvd_prediction_user_male")
        height = tracker.get_slot("height")
        weight = tracker.get_slot("weight")
        systolic_blood_pressure = int(tracker.get_slot("systolic_blood_pressure"))
        diastolic_blood_pressure = int(tracker.get_slot("diastolic_blood_pressure"))
        blood_pressure_type= get_blood_pressure_type(systolic_blood_pressure, diastolic_blood_pressure)
        marker_blood_pressure=f"marker_cvd_prediction_user_has_{blood_pressure_type}_bp"
        save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker=marker_blood_pressure)
        glucose_level = tracker.get_slot("glucose_level")
        cholesterol_level = tracker.get_slot("cholesterol_level")
        drinks_alcohol = tracker.get_slot("drinks_alcohol")
        if(drinks_alcohol == 'True'):
            # add_marker_to_csv(sender_id,session_idx, 'marker_cvd_prediction_user_drinks_alcohol')
            save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain,
                            marker="marker_cvd_prediction_user_drinks_alcohol")
        is_smoking = tracker.get_slot("is_smoking")
        if(is_smoking == 'True'):
            # add_marker_to_csv(sender_id,session_idx, 'marker_cvd_prediction_user_is_smoking')
            save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain,
                            marker="marker_cvd_prediction_user_is_smoking")
        is_physically_active = tracker.get_slot("is_physically_active")
        if (is_physically_active == 'True'):
            # add_marker_to_csv(sender_id,session_idx, 'marker_cvd_prediction_user_is_physically_active')
            save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain,
                            marker="marker_cvd_prediction_user_is_physically_active")
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
            # add_marker_to_csv(sender_id, session_idx, 'marker_cvd_prediction_true')
            save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_cvd_prediction_true")
            dispatcher.utter_message("You are at risk of getting a cardiovascular disease. Please take care of yourself and just do regular check up of yours.")
        else:
            # add_marker_to_csv(sender_id, session_idx, 'marker_cvd_prediction_false')
            save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_cvd_prediction_false")
            dispatcher.utter_message("Congratulations, you are out of risk of getting any cardiovascular disease.")

        # statistics.compute_statistics_gender_ratio()
        # statistics.compute_statistics_users_habits()
        # statistics.compute_statistics_users_blood_pressure()
        # statistics.compute_statistics_cvd_prediction_ratio()
        # statistics.compute_statistics_user_responses_by_month()
        # stats = ActionComputeStatistics()
        # stats.run(dispatcher=dispatcher, tracker=tracker, domain=domain)


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
            return ""
    def give_recommendation_user_drinks_alcohol(self, drinks_alcohol):
        if(drinks_alcohol=='True'):
            advice="Drinking too much alcohol can raise blood pressure levels and the risk for heart disease. \n" \
                   "It also increases levels of triglycerides, a fatty substance in the blood which can increase the risk for heart disease. \n" \
                   "If you drink alcohol, try not to exceed the recommended limit of 14 alcohol units a week."
            return advice
        else:
            return ""
    def give_recommendation_user_is_physically_active(self, is_physically_active):
        advice = ""
        if (is_physically_active == 'False'):
            advice = "If you don't exercise regularly, it's more likely that you'll have high blood pressure, high cholesterol levels and be overweight." \
                     "All of these are risk factors for CVD. \n" \
                     "Exercising regularly will help keep your heart healthy. When combined with a healthy diet, exercise can also help you maintain a healthy weight."
        return advice

    # https: // www.apollo247.com / blog / article / health - risks - being - underweight
    def give_recommendation_user_bmi(self, height, weight):
        advice=""
        bmi = get_bmi(height, weight)
        if(bmi >= 25):
            advice=f"Since your BMI (Body Mass Index) is {bmi}, you are at an increased risk of a heart disease. \n" \
                   f"Being overweight is generally caused by consuming more calories, particularly those in fatty and sugary foods, than you burn off through physical activity. \n" \
                   f"You should aim to get your BMI below 25. \n" \
                   f"You can achieve this by eating a balanced calorie-controlled diet and exercise regularly." \
                   f"Also eat slowly and avoid situations where you know you could be tempted to overeat. \n" \
                   f"If you're struggling to lose weight, your GP or practice nurse can help you come up with a weight loss plan and recommend services in your area."
        elif (bmi <= 18.5):
            advice=f"Since your BMI (Body Mass Index) is {bmi}, you seem to be underweight. \n" \
                   f" While obesity increases the risk of developing several health issues, being underweight is equally serious. You should consult a doctor if you experience symptoms such as persistent tiredness, loss of appetite, and increased incidences of illnesses and infections. \n" \
                   f"Studies have shown that underweight people suffer from nutritional deficiencies, which can increase your risk of developing heart diseases such as as mitral valve prolapse (bulging of the valves of the heart), arrhythmias, and even heart failure, gradual bone loss, reduced immune function, anaemia and other diseases. \n" \
                   f"Make sure to adopt a healthy diet which should include minerals such as potassium, sodium, and calcium, which are required to maintain a steady heartbeat, " \
                   f"and aim to get your BMI between 18.5 and 25."
        return advice

    def give_recommendation_user_age(self, age):
        advice=""
        age_bin = categorize_age_bin(age)
        if(age_bin >= 4):
            advice="Please know that cardiovascular diseases are most common in people over 50 and your risk of developing them increases as you get older."

        return advice
    # def give_recommendation_user_systolic_blood_pressure(self):
    #     pass
    # def give_recommendation_user_diastolic_blood_pressure(self):
    #     pass
    # https: // www.webmd.com / heart / understanding - low - blood - pressure - treatment
    # https: // www.mayoclinic.org / diseases - conditions / prehypertension / diagnosis - treatment / drc - 20376708
    def give_recommendation_user_blood_pressure(self, blood_pressure_type):
        advice = ""
        if (blood_pressure_type == "low"):
            advice = "Since you have a low blood pressure, please make sure" \
                     "to eat a diet higher in salt, drink lots of nonalcoholic fluids," \
                     "check with your doctor to see if any medication you take are causing this," \
                     "try eating smaller, more frequent meals and, if needed, use compression stockings to keep more blood in the upper body"
        elif (blood_pressure_type == "normal"):
            advice = "Your blood pressure is normal, so please keep up a healthy lifestyle."
        elif (blood_pressure_type == "elevated"):
            advice= "As your blood pressure level is elevated, you need to change your lifestyle in order to reduce and keep it under control." \
                    "Please make sure to eat healthy foods, maintain a healthy weight," \
                    "use less salt in your diet, increase your physical activity and manage stress."
        else:
            advice="Your blood pressure levels indicate that you have hypertension." \
                   "Your doctor can help you keep your blood pressure to a safe level using medicines and lifestyle changes." \
                   "Talk to your doctor to help you decide about treatment." \
                   ""
        return advice

    # https: // www.nhs.uk / conditions / high - blood - sugar - hyperglycaemia /
    def give_recommendation_user_glucose_level(self, glucose_level):
        advice=""
        glucose = get_glucose_level(glucose_level)
        if(glucose >=2):
            advice="Hyperglycemia, or high blood glucose, occurs when there is too much sugar in the blood. \n" \
                   "In order to lower your glucose level, you should: \n" \
                   "change your diet, try to avoid foods that cause your blood sugar levels to rise, such as cakes or sugary drinks \n" \
                   "drink plenty of sugar-free fluids – this can help if you're dehydrated \n" \
                   "exercise more often – gentle, regular exercise such as walking can often lower your blood sugar level, particularly if it helps you lose weight \n" \
                   "if you use insulin, adjust your dose – your doctor can give you specific advice about how to do this \n" \
                   "Until your blood sugar level is back under control, watch out for additional symptoms that could be a sign of a more serious condition."
        return advice
    def give_recommendation_user_cholesterol_level(self, cholesterol_level):
        advice=""
        cholesterol = get_cholesterol_level(cholesterol_level)
        if (cholesterol >= 2):
            advice="Besides stopping smoking and cutting down on alcohol, what else you can do to lower your cholesterol level is to " \
                   "eat less fatty food, especially especially food that contains a type of fat called saturated fat." \
                   "You should try to eat less fatty meat, butter, cream and hard cheese, cakes and biscuits and food that contains coconut oil or palm oil." \
                   "Additionally, you should aim to do at least 150 minutes (2.5 hours) of exercise a week."
        return advice
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        sender_id = tracker.current_state()['sender_id']

        session_idx = tracker.idx_after_latest_restart()

        # add_marker_to_csv(sender_id,session_idx,'marker_asked_for_advice')
        save_marker.run(dispatcher=dispatcher, tracker=tracker, domain=domain, marker="marker_asked_for_advice")
        age = tracker.get_slot("age")
        gender = tracker.get_slot("gender")
        height = tracker.get_slot("height")
        weight = tracker.get_slot("weight")
        systolic_blood_pressure = int(tracker.get_slot("systolic_blood_pressure"))
        diastolic_blood_pressure = int(tracker.get_slot("diastolic_blood_pressure"))
        blood_pressure_type= get_blood_pressure_type(systolic_blood_pressure, diastolic_blood_pressure)
        glucose_level = tracker.get_slot("glucose_level")
        cholesterol_level = tracker.get_slot("cholesterol_level")
        drinks_alcohol = tracker.get_slot("drinks_alcohol")
        is_smoking = tracker.get_slot("is_smoking")
        is_physically_active = tracker.get_slot("is_physically_active")


        # bmi=calculate_bmi(height, weight)

        dispatcher.utter_message(self.give_recommendation_user_bmi(height, weight))
        dispatcher.utter_message(self.give_recommendation_user_blood_pressure(blood_pressure_type))
        dispatcher.utter_message(self.give_recommendation_user_is_smoking(is_smoking))
        dispatcher.utter_message(self.give_recommendation_user_cholesterol_level(cholesterol_level))
        dispatcher.utter_message(self.give_recommendation_user_glucose_level(glucose_level))
        dispatcher.utter_message(self.give_recommendation_user_drinks_alcohol(drinks_alcohol))
        dispatcher.utter_message(self.give_recommendation_user_is_physically_active(is_physically_active))


        return []


class ActionAnswerQuestion(Action):
    # document_store = InMemoryDocumentStore()
    # document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", sql_url="sqlite:///faiss_document_store.db")
    document_store = FAISSDocumentStore.load(index_path="testfile_path.index")
    # topics = [36808, 57330, 3997, 512662, 625404, 249930, 60575]
    # retriever = TfidfRetriever(document_store=document_store)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers"
    )

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", num_processes=1)
    pipe = ExtractiveQAPipeline(reader, retriever)
    # data_dir = "D:/BachelorThesis/rasa-chatbot/actions/data"

    def __init__(self) -> None:
        logger.debug(f"Creating {self.name()} custom action ...")
        # logger.info(f"Saving wikipedia articles to {self.data_dir}")
        # get_wikipedia_articles(self.data_dir, topics=self.topics)

        logger.info("Converting wikipedia articles to documents ...")
    #     docs = convert_files_to_dicts(
    #     dir_path=self.data_dir, clean_func=clean_wiki_text, split_paragraphs=True
    # )

        logger.info("Writing documents to document store")
        # self.document_store.write_documents(docs)
        # self.document_store.update_embeddings(self.retriever)
        # self.document_store.save("testfile_path.index")



    def _get_answer(self, question):
        prediction = self.pipe.run(
            query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )
        return prediction['answers'][0].answer

    def name(self) -> Text:
        return "action_answer_question"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        question = tracker.latest_message["text"]
        print("Type:")
        print(type(question))
        print(question)
        print("Done!!")
        r = self._get_answer(question)
        cleaned_answer = re.sub("[@#$^~_+€™âăîțș]", "", r)
        dispatcher.utter_message(text=cleaned_answer)

        return []






class ActionCalculateBP(Action):
    def name(self) -> Text:
        return "action_calculate_bp"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        systolic_blood_pressure = int(tracker.get_slot("systolic_blood_pressure"))
        diastolic_blood_pressure = int(tracker.get_slot("diastolic_blood_pressure"))
        blood_pressure_type = get_blood_pressure_type(systolic_blood_pressure, diastolic_blood_pressure)
        dispatcher.utter_message(text=f"Your blood pressure falls into {blood_pressure_type} category")

        return []