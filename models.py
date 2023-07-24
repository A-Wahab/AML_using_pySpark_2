import os
import pickle
import db_conn
import numpy as np
import pandas as pd
from statistics import mode
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class Model:

    def __init__(self, dataset_name, task):
        self.classifiers = []
        self.regressors = []
        self.model_dic = {}
        self.dataset_name = dataset_name
        self.task = task
        self.trained_models = []
        self.initialize()
        self.path = 'Models/' + db_conn.current_user_session() + '/' + self.task + '/' + self.dataset_name

    def initialize(self):
        self.classifiers = ['Logistic Regression', 'Naive Bayes', 'Random Forest Classifier',
                            'Support Vector Classifier', 'K Nearest Neighbour Classifier']

        self.regressors = ['Linear Regression', 'Random Forest Regressor',
                           'Support Vector Regressor', 'K Nearest Neighbour Regressor']

        classifier_models = [LogisticRegression(), GaussianNB(), RandomForestClassifier(), SVC(),
                             KNeighborsClassifier()]
        regressor_models = [LinearRegression(), RandomForestRegressor(), SVR(), KNeighborsRegressor()]

        self.model_dic = {key: value for key, value in zip(self.classifiers, classifier_models)}
        self.model_dic.update({key: value for key, value in zip(self.regressors, regressor_models)})

    def train(self, input_features, output_feature, models):
        self.trained_models = []
        for model in models:
            model = self.model_dic.get(model)
            model.fit(input_features, output_feature)
            self.trained_models.append(model)

    def predict(self, unclassified_instances, task):

        predictions = pd.DataFrame()
        for model in self.load_models():
            predictions = pd.concat([predictions, pd.DataFrame(model.predict(unclassified_instances))], axis=1)

        if task == 'Classification':
            return [mode(x) for x in predictions.values]

        return [np.mean(x) for x in predictions.values]

    def save_models(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for model in self.trained_models:
            pickle.dump(model, open(self.path + '/' + str(model)[:-2] + '.pkl', 'wb'))

    def load_models(self):
        models = []
        saved_models = os.listdir(self.path)
        for model in saved_models:
            models.append(pickle.load(open(self.path + '/' + model, 'rb')))

        return models

    @staticmethod
    def check_trained_datasets(dataset, task):
        return dataset in os.listdir('Models/' + db_conn.current_user_session() + '/' + task + '/')
