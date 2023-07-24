import sqlite3
import db_conn
import pandas as pd
import preprocessing
import streamlit as st
from models import Model
import dummy_data_generator

permits = db_conn.get_permits()
conn = sqlite3.connect('db/database.db')
cursor = conn.cursor()


def some_credentials():
    return st.text_input('Enter the name of your data set'), st.selectbox('Tell us about your problem',
                                                                          ['Classification', 'Regression'])


def start():
    try:
        st.title('Voila !\nTired of doing all the same procedure of ML over and over and over again ?\n'
                 '\nWell, we got you! You came to the right place baby !')

        option = st.selectbox('What you wanna do ?', ['Train', 'Predict',
                                                      'Generate your own dummy dataset'])

        name, type_of_prediction = some_credentials()
        models = Model(dataset_name=name, task=type_of_prediction)

        if option == 'Train':

            file = st.file_uploader('upload your dataset')

            if file is not None:
                df = pd.read_csv(file)
                target_variable = st.selectbox('Select the target variable', df.columns)
                df = preprocessing.preprocess(
                    pd.concat([df.drop(target_variable, axis=1), df[target_variable]], axis=1))
                st.write(df)

                if type_of_prediction == 'Classification':
                    model = st.multiselect('Choose your classifier(s)', models.classifiers,
                                           default='Random Forest Classifier')
                else:
                    model = st.multiselect('Choose your regressor(s)', models.regressors,
                                           default='Random Forest Regressor')

                x_train, x_test, y_train, y_test = preprocessing.split(df.drop(target_variable, axis=1),
                                                                       df[target_variable])
                models.train(x_train, y_train, models=model)
                models.save_models()
                predictions = models.predict(x_test, task=type_of_prediction)
                st.write(predictions)

        if option == 'Predict':
            if models.check_trained_datasets(name, type_of_prediction):
                file = st.file_uploader('provide a dataset')
                if file is not None:
                    df = pd.read_csv(file)
                    predictions = models.predict(df, task=type_of_prediction)
                    pd.DataFrame(predictions).to_csv('file.csv')
                    st.write(pd.concat([df, pd.DataFrame({
                        'Predictions': predictions
                    })], axis=1))
            else:
                st.write(
                    'We do not have any pre-trained model for this dataset. We recommend training a model(s) on this '
                    'dataset first.\nThank You!')

        if option == 'Generate your own dummy dataset':
            dummy_data = dummy_data_generator.generate(name)
            st.write(dummy_data)
            st.download_button('Download your generated data',
                               data=dummy_data.to_csv().encode('utf-8'), mime='csv', file_name=name + '.csv')

        if st.button('sign out'):
            db_conn.sign_out()
    except:
        pass


if permits['log']:
    if db_conn.log_in():
        db_conn.update_permits(log=False, reg=False, start=True)
        st.experimental_rerun()

elif permits['reg']:
    db_conn.register()

elif permits['start']:
    start()
