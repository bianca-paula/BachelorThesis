from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np

data = pd.read_csv("markers_values.csv", sep=",")
# print(dataset)
# print(dataset.info())

count_asked_for_advice = data[data["marker"]=="marker_asked_for_advice"].shape[0]
count_found_advice_useful= data[data["marker"]=="marker_found_advice_useful"].shape[0]
count_asked_for_cvd_prediction = data[data["marker"]=="marker_asked_for_cvd_prediction"].shape[0]
count_user_began_conversation = data[data["marker"]=="marker_user_began_conversation"].shape[0]
count_user_asked_faq = data[data["marker"]=="marker_user_asked_faq"].shape[0]
count_number_of_females = data[data["marker"]=="marker_cvd_prediction_user_female"].shape[0]
count_number_of_males = data[data["marker"]=="marker_cvd_prediction_user_male"].shape[0]
count_cvd_prediction_false = data[data["marker"]=="marker_cvd_prediction_false"].shape[0]
count_cvd_prediction_true = data[data["marker"]=="marker_cvd_prediction_true"].shape[0]
count_user_drinks_alcohol = data[data["marker"]=="marker_cvd_prediction_user_drinks_alcohol"].shape[0]
count_user_is_smoking = data[data["marker"]=="marker_cvd_prediction_user_is_smoking"].shape[0]
count_user_is_active = data[data["marker"]=="marker_cvd_prediction_user_is_physically_active"].shape[0]

count_user_has_normal_bp = data[data["marker"]=="marker_cvd_prediction_user_has_normal_bp"].shape[0]
count_user_has_elevated_bp = data[data["marker"]=="marker_cvd_prediction_user_has_elevated_bp"].shape[0]
count_user_has_hypertension_bp = data[data["marker"]=="marker_cvd_prediction_user_has_hypertension_bp"].shape[0]

labels = ['females', 'males']
sizes = [count_number_of_females, count_number_of_males]

# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()
#
#
#
# # Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'At risk', 'Not at risk'
# sizes = [count_cvd_prediction_true, count_cvd_prediction_false]
# explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()


#
# labels = ['Smoking', 'Drinking', 'Active']
# no_means = [count_asked_for_cvd_prediction-count_user_is_smoking, count_asked_for_cvd_prediction-count_user_drinks_alcohol, count_asked_for_cvd_prediction-count_user_is_active]
# yes_means = [count_user_is_smoking, count_user_drinks_alcohol, count_user_is_active]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x + width/2, no_means, width, label='No')
# rects2 = ax.bar(x - width/2, yes_means, width, label='Yes')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('User habits')
# ax.set_xticks(x, labels)
# ax.legend()
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
#
#
# fig.tight_layout()
#
# plt.show()
# # plt.savefig('foo.png')


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'Normal', 'Elevated', "Hypertension"
# sizes = [count_user_has_normal_bp, count_user_has_elevated_bp, count_user_has_hypertension_bp]
# explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()


current_year = datetime.now().year
print(current_year)

current_month = datetime.now().month
print(current_month)



data['timestamp']=pd.to_datetime(data['timestamp'], format="%Y-%m-%d %H:%M:%S")
# print(data[(data['timestamp'].dt.day==27) & (data['timestamp'].dt.year==2022) & (data['timestamp'].dt.month==4)])


# responses=[]
# months = list(range(1, current_month+1))
# months = np.arange(1, current_month+1)
# print(months)
# # responses=np.empty(shape=1)
# print(responses)
#
# for i in range(1, current_month+1):
#     current_month_responses=data[(data['timestamp'].dt.year==current_year) & (data['timestamp'].dt.month==i) & (data['marker']== 'marker_asked_for_cvd_prediction')].shape[0]
#     responses.append(current_month_responses)
#
#
# print(data.info())
#
# print(months)
# print(responses)
#
# responses_array = np.asarray(responses)
#
# fig = plt.figure(figsize=(10,5))
#
# plt.plot(months, responses_array)
# plt.xticks(months)
# plt.xlim(1, np.amax(months))
#
# plt.ylim(0)
#
# plt.fill_between(months, responses_array)
# plt.xlabel("Month")
# plt.ylabel("Number of responses")
# plt.title("Number of entries")
#
# plt.show()







# fig, ax = plt.subplots(figsize=(6, 6))
# wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':3}
# ax.pie([count_found_advice_useful,count_asked_for_advice-count_found_advice_useful], wedgeprops=wedgeprops, startangle=90, colors=['#5DADE2', '#515A5A'])
# plt.title('Users found advice useful', fontsize=24, loc='left')
# percentage = round((count_found_advice_useful/count_asked_for_advice)*100, 1)
# plt.text(0, 0, str(percentage)+"%", ha='center', va='center', fontsize=42)
# plt.show()


# fig, ax = plt.subplots(figsize=(6, 6))
# wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':3}
# ax.pie([count_asked_for_cvd_prediction,count_user_began_conversation-count_asked_for_cvd_prediction], wedgeprops=wedgeprops, startangle=90, colors=['#5DADE2', '#515A5A'])
# plt.title('Users asked for CVD prediction', fontsize=24, loc='left')
# percentage = round((count_asked_for_cvd_prediction/count_user_began_conversation)*100, 1)
# plt.text(0, 0, str(percentage)+"%", ha='center', va='center', fontsize=42)
# plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':3}
ax.pie([count_user_asked_faq,count_user_began_conversation-count_user_asked_faq], wedgeprops=wedgeprops, startangle=90, colors=['#5DADE2', '#515A5A'])
plt.title('Users asked FAQ', fontsize=24, loc='left')
percentage = round((count_user_asked_faq/count_user_began_conversation)*100, 1)
plt.text(0, 0, str(percentage)+"%", ha='center', va='center', fontsize=42)
plt.show()

