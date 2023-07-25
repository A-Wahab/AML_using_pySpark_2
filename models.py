import os
import pickle
import db_conn
import numpy as np
import pandas as pd
import streamlit as st
from statistics import mode
from pyspark.sql import SparkSession
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from pyspark.ml.feature import VectorAssembler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pyspark.sql.types import StringType, StructType, StructField


class Model:

    def __init__(self, dataset_name, task):
        self.spark = SparkSession.builder.appName("ModelTraining").getOrCreate()
        self.classifiers = []
        self.regressors = []
        self.model_dic = {}
        self.dataset_name = dataset_name
        self.task = task
        self.trained_models = []
        self.initialize()
        self.selected_models = None
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

    def go(self, df, target_variable, models):
        self.selected_models = models
        train, test = self.split(df, target_variable)
        self.train(train, target_variable, models=models)
        self.save_models()
        predictions = self.predict(test, self.task, fun='evaluate')

    @staticmethod
    def split(df, target_variable):
        input_cols = df.columns[-1]
        assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
        assembled_data = assembler.transform(df).select("features", target_variable)
        return assembled_data.randomSplit([0.8, 0.2], seed=42)

    def train(self, train, output_feature, models):
        self.trained_models = []
        for model in models:
            model = self.model_dic.get(model)
            model.setFeatureCol("features")
            model.setLabelCol(output_feature)
            model.fit(train)
            self.trained_models.append(model)

    def predict(self, unclassified_instances, task, fun):
        i = 0
        trained_models = self.load_models()
        schema = StructType([StructField(col, StringType(), nullable=True) for col in trained_models])
        all_predictions = self.spark.createDataFrame(data=[], schema=schema)

        for model in trained_models:
            predictions = getattr(model, fun)(unclassified_instances).predictions.select('prediction')
            all_predictions.withColumn(self.selected_models[i], predictions)
            i = i + 1

        st.write(all_predictions)
        # if task == 'Classification':
        #     return [mode(x) for x in predictions.values]
        #
        # return [np.mean(x) for x in predictions.values]

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
