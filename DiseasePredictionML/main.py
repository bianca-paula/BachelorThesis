import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#https://www.geeksforgeeks.org/disease-prediction-using-machine-learning/
#dataset: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

# The path to the dataset file that we will use for training ("Training.csv"), located
# under the dataset folder
TRAINING_DATA_FILEPATH = "dataset/Training.csv"
training_data = pd.read_csv(TRAINING_DATA_FILEPATH).dropna(axis=1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
