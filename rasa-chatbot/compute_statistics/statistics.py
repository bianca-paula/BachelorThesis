import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib import rcParams
import numpy as np

class Statistics:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path, sep=",")

    def get_count_asked_for_advice(self):
        count_asked_for_advice = self.data[self.data["marker"] == "marker_asked_for_advice"].shape[0]
        return count_asked_for_advice
    def get_count_found_advice_useful(self):
        count_found_advice_useful= self.data[self.data["marker"]=="marker_found_advice_useful"].shape[0]
        return count_found_advice_useful
    def get_count_asked_for_cvd_prediction(self):
        count_asked_for_cvd_prediction = self.data[self.data["marker"]=="marker_asked_for_cvd_prediction"].shape[0]
        return count_asked_for_cvd_prediction
    def get_count_user_began_conversation(self):
        count_user_began_conversation = self.data[self.data["marker"]=="marker_user_began_conversation"].shape[0]
        return count_user_began_conversation
    def get_count_user_asked_faq(self):
        count_user_asked_faq = self.data[self.data["marker"]=="marker_user_asked_faq"].shape[0]
        return count_user_asked_faq
    def get_count_number_of_females(self):
        count_number_of_females = self.data[self.data["marker"]=="marker_cvd_prediction_user_female"].shape[0]
        return count_number_of_females
    def get_count_number_of_males(self):
        count_number_of_males = self.data[self.data["marker"]=="marker_cvd_prediction_user_male"].shape[0]
        return count_number_of_males
    def get_count_cvd_prediction_false(self):
        count_cvd_prediction_false = self.data[self.data["marker"]=="marker_cvd_prediction_false"].shape[0]
        return count_cvd_prediction_false
    def get_count_cvd_prediction_true(self):
        count_cvd_prediction_true = self.data[self.data["marker"]=="marker_cvd_prediction_true"].shape[0]
        return count_cvd_prediction_true
    def get_count_user_drinks_alcohol(self):
        count_user_drinks_alcohol = self.data[self.data["marker"]=="marker_cvd_prediction_user_drinks_alcohol"].shape[0]
        return count_user_drinks_alcohol
    def get_count_user_is_smoking(self):
        count_user_is_smoking = self.data[self.data["marker"]=="marker_cvd_prediction_user_is_smoking"].shape[0]
        return count_user_is_smoking
    def get_count_user_is_active(self):
        count_user_is_active = self.data[self.data["marker"]=="marker_cvd_prediction_user_is_physically_active"].shape[0]
        return count_user_is_active
    def get_count_user_has_low_bp(self):
        count_user_has_low_bp = self.data[self.data["marker"]=="marker_cvd_prediction_user_has_low_bp"].shape[0]
        return count_user_has_low_bp
    def get_count_user_has_normal_bp(self):
        count_user_has_normal_bp = self.data[self.data["marker"]=="marker_cvd_prediction_user_has_normal_bp"].shape[0]
        return count_user_has_normal_bp
    def get_count_user_has_elevated_bp(self):
        count_user_has_elevated_bp = self.data[self.data["marker"]=="marker_cvd_prediction_user_has_elevated_bp"].shape[0]
        return count_user_has_elevated_bp
    def get_count_user_has_hypertension_bp(self):
        count_user_has_hypertension_bp = self.data[self.data["marker"]=="marker_cvd_prediction_user_has_hypertension_bp"].shape[0]
        return count_user_has_hypertension_bp

    def compute_statistics_gender_ratio(self):
        labels = ['females', 'males']
        sizes = [self.get_count_number_of_females(), self.get_count_number_of_males()]

        figure, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # plt.show()
        plt.savefig('./compute_statistics/statistics_images/gender_ratio.png')

    def compute_statistics_cvd_prediction_ratio(self):
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'At risk', 'Not at risk'
        sizes = [self.get_count_cvd_prediction_true(), self.get_count_cvd_prediction_false()]
        explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # plt.show()
        plt.savefig('./compute_statistics/statistics_images/cvd_prediction_ratio.png')


    def compute_statistics_users_habits(self):
        labels = ['Smoking', 'Drinking', 'Active']
        no_means = [self.get_count_asked_for_cvd_prediction()-self.get_count_user_is_smoking(), self.get_count_asked_for_cvd_prediction()-self.get_count_user_drinks_alcohol(), self.get_count_asked_for_cvd_prediction()-self.get_count_user_is_active()]
        yes_means = [self.get_count_user_is_smoking(), self.get_count_user_drinks_alcohol(), self.get_count_user_is_active()]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x + width/2, no_means, width, label='No')
        rects2 = ax.bar(x - width/2, yes_means, width, label='Yes')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('User habits')
        ax.set_xticks(x, labels)
        ax.legend()
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)


        fig.tight_layout()

        # plt.show()
        plt.savefig('./compute_statistics/statistics_images/user_habits.png')
    def compute_statistics_users_blood_pressure(self):
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Low', 'Normal', 'Elevated', "Hypertension"
        sizes = [self.get_count_user_has_low_bp(), self.get_count_user_has_normal_bp(),self.get_count_user_has_elevated_bp(), self.get_count_user_has_hypertension_bp()]
        explode = (0, 0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # plt.show()
        plt.savefig('./compute_statistics/statistics_images/users_blood_pressure.png')


    def compute_statistics_user_responses_by_month(self):
        current_year = datetime.now().year
        current_month = datetime.now().month
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format="%Y-%m-%d %H:%M:%S")
        responses=[]
        months = list(range(1, current_month+1))
        months = np.arange(1, current_month+1)
        print(months)
        # responses=np.empty(shape=1)
        print(responses)

        for i in range(1, current_month+1):
            current_month_responses=self.data[(self.data['timestamp'].dt.year==current_year) & (self.data['timestamp'].dt.month==i) & (self.data['marker']== 'marker_asked_for_cvd_prediction')].shape[0]
            responses.append(current_month_responses)


        print(months)
        print(responses)

        responses_array = np.asarray(responses)

        fig = plt.figure(figsize=(10,5))

        plt.plot(months, responses_array)
        plt.xticks(months)
        plt.xlim(1, np.amax(months))

        plt.ylim(0)

        plt.fill_between(months, responses_array)
        plt.xlabel("Month")
        plt.ylabel("Number of responses")
        plt.title("Number of entries")

        # plt.show()
        plt.savefig('./compute_statistics/statistics_images/graph_queries_by_month.png')


    def compute_statistics_users_found_advice_useful(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':3}
        ax.pie([self.get_count_found_advice_useful(),self.get_count_asked_for_advice()-self.get_count_found_advice_useful()], wedgeprops=wedgeprops, startangle=90, colors=['#5DADE2', '#515A5A'])
        plt.title('Users found advice useful', fontsize=24, loc='left')
        percentage = round((self.get_count_found_advice_useful()/self.get_count_asked_for_advice())*100, 1)
        plt.text(0, 0, str(percentage)+"%", ha='center', va='center', fontsize=42)
        # plt.show()
        plt.savefig('./compute_statistics/statistics_images/users_found_advice_useful.png')


    def compute_statistics_users_asked_for_cvd_prediction(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':3}
        ax.pie([self.get_count_asked_for_cvd_prediction(), self.get_count_user_began_conversation()-self.get_count_asked_for_cvd_prediction()], wedgeprops=wedgeprops, startangle=90, colors=['#5DADE2', '#515A5A'])
        plt.title('Users asked for CVD prediction', fontsize=24, loc='left')
        percentage = round((self.get_count_asked_for_cvd_prediction()/self.get_count_user_began_conversation())*100, 1)
        plt.text(0, 0, str(percentage)+"%", ha='center', va='center', fontsize=42)
        # plt.show()
        plt.savefig('./compute_statistics/statistics_images/users_asked_for_cvd_prediction.png')


    def compute_statistics_users_asked_faq(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        wedgeprops = {'width': 0.3, 'edgecolor': 'black', 'linewidth': 3}
        ax.pie([self.get_count_user_asked_faq(), self.get_count_user_began_conversation() - self.get_count_user_asked_faq()], wedgeprops=wedgeprops,
               startangle=90, colors=['#5DADE2', '#515A5A'])
        plt.title('Users asked FAQ', fontsize=24, loc='left')
        percentage = round((self.get_count_user_asked_faq() / self.get_count_user_began_conversation()) * 100, 1)
        plt.text(0, 0, str(percentage) + "%", ha='center', va='center', fontsize=42)
        # plt.show()
        plt.savefig('./compute_statistics/statistics_images/users_asked_faq.png')



#
# s= Statistics(path="../markers_values.csv")
#
# s.compute_statistics_gender_ratio()
# s.compute_statistics_cvd_prediction_ratio()
# s.compute_statistics_users_habits()
# s.compute_statistics_users_blood_pressure()
# s.compute_statistics_user_responses_by_month()
# s.compute_statistics_users_asked_faq()
# s.compute_statistics_users_asked_for_cvd_prediction()
# s.compute_statistics_users_found_advice_useful()
