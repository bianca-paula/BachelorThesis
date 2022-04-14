import pandas as pd


class DataLoader:
    def __init__(self, filepath):
        # read the dataset from the csv file
        dataset = pd.read_csv(filepath, sep=";")

    def remove_height_outliers(self, dataset):
        dataset.drop(dataset[(dataset['height'] > dataset['height'].quantile(0.975)) | (dataset['height'] < dataset['height'].quantile(0.024))].index, inplace=True)
    def remove_weight_outliers(self, dataset):
        dataset.drop(dataset[(dataset['weight'] > dataset['weight'].quantile(0.975)) | (dataset['weight'] < dataset['weight'].quantile(0.024))].index, inplace=True)
    def remove_diastolic_pressure_outliers(self, dataset):
        dataset.drop(dataset[(dataset['ap_lo'] > dataset['ap_lo'].quantile(0.98)) | (dataset['ap_lo'] < dataset['ap_lo'].quantile(0.019))].index, inplace=True)
    def remove_systolic_pressure_outliers(self, dataset):
        dataset.drop(dataset[(dataset['ap_hi'] > dataset['ap_hi'].quantile(0.98)) | (dataset['ap_hi'] < dataset['ap_hi'].quantile(0.019))].index, inplace=True)

    def convert_height_cm_to_m(self, dataset):
        dataset.height = dataset['height'] / 100

    def convert_age_to_years(self, dataset):
        dataset['years'] = (dataset['age'] / 365).round().astype('int')
        dataset.age = dataset.years
        dataset = dataset.drop(['years'], axis=1)
    def drop_id_column(self, dataset):
        dataset = dataset.drop(['id'], axis=1)

    def create_age_bin(self, dataset):
        dataset['age_bin'] = pd.cut(dataset['age'],
                                    [0, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                                    labels=['0-20', '20-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60',
                                            '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100'])
    def add_body_mass_index(self, dataset):
        dataset['bmi'] = dataset['weight'] / ((dataset['height'] / 100) ** 2)
        rating = []
        for row in dataset['bmi']:
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
        dataset['BMI_Class'] = rating
    def add_MAP_column(self, dataset):
        # creating a Column for MAP
        dataset['MAP'] = ((2 * dataset['ap_lo']) + dataset['ap_hi']) / 3
        # Creating Classes for MAP
        map_values = []
        for row in dataset['MAP']:
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

        #inserting MAP_Class Column
        dataset['MAP_Class'] = map_values

    def reorder_columns(self, dataset):
        dataset = dataset[
            ["gender", "height", "weight", "bmi", "ap_hi", "ap_lo", "MAP", "age", "age_bin", "BMI_Class", "MAP_Class",
             "cholesterol", "gluc", "smoke", "active", "cardio"]]
