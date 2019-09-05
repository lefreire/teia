import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


class PreProcessing:

    def __init__(self, path):
        self.path = path
        self.data = None

    def load_dataset(self):
        #load Dataset
        self.data = pd.read_csv(self.path)
        return self.data

    def translate_columns(self):
        for column in self.data:
            if self.data[column].dtype == 'O':
                self.data = self.translate_text(column)
            else:
                self.data = self.fill_nan_values(column)

    def translate_text(self, column):
        columns = self.data[column].unique().tolist()
        dict_col = {}
        for col in range(0, len(columns)):
            dict_col[columns[col]] = col

        is_nan = self.data[column].isna()     

        for row in range(0, len(self.data[column])):
            if is_nan[row]:
                self.data[column][row] = np.random.randint(0, len(columns))
            else:
                self.data[column][row] = dict_col[self.data[column][row]]
        return self.data

    def fill_nan_values(self, column):
        #when the column is float or int
        is_nan = self.data[column].isna() 
        all_num = [self.data[column][i] for i in 
                        range(0, len(self.data[column])) if is_nan[i] == False] 
        
        mean_column = np.mean(all_num)
        std = np.std(mean_column)

        for row in range(0, len(self.data[column])):
            if is_nan[row]:
                self.data[column][row] = np.random.normal(mean_column, np.abs(std))
        return self.data

    def save_file(self, name):
        self.data.to_csv(name, index=False)

    def separate_data(self, data, label_name='target'):
        data_full = self.data.copy()
        X_data = data_full.drop(label_name, axis=1)
        y = data_full[label_name]
        return X_data, y

    def split_train_test(self, data, label_name='target'):
        X_data, y = self.separate_data(data, label_name)
        test_size = 0.3
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, 
            test_size=test_size)
        return X_train, X_test, y_train, y_test 

    def copy_columns(self, columns, label_name, X_train, X_test):
        #columns = vetor of 0 and 1, indicating if a column will be part of the dataset
        data_columns = self.data.columns.tolist()
        selected_columns = []
        for column in range(0, len(data_columns)):
            if columns[column]: selected_columns.append(data_columns[column])
        if len(selected_columns) == 0: 
            selected_columns.append(data_columns[0])
        if label_name in selected_columns: selected_columns.remove(label_name)
        return X_train[selected_columns], X_test[selected_columns]