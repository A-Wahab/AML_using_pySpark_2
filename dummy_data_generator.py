import pandas as pd
import streamlit as st
import numpy as np


def generate(dataset_name):
    data = ask_for_preferences(dataset_name=dataset_name)
    return data


def ask_for_preferences(dataset_name):
    total_samples = st.number_input('Number of samples to be generated', min_value=1, step=1, value=10)
    cont_feature_names = st.text_input('Write your continues input features names '
                                       '(e.g. Age, Weight, Height)').split(',')
    disc_feature_names = st.text_input('Write your discrete input features names '
                                       '(e.g. Gender, Section, Grade)').split(',')

    dummy_cont_data = generate_cont_data(total_samples, cont_feature_names)
    dummy_disc_data = generate_disc_data(total_samples, disc_feature_names)

    input_data = pd.concat([dummy_cont_data, dummy_disc_data], axis=1)

    output_feature_name = st.text_input("Write your output feature / target variable's name")
    dtype_output_feature = st.selectbox('What should be the type of the output feature ?',
                                        options=['Continues', 'Discrete'])

    if dtype_output_feature == 'Continues':
        output_data = generate_cont_data(total_samples, [output_feature_name])
    else:
        output_data = generate_disc_data(total_samples, [output_feature_name])

    final_df = pd.concat([input_data, output_data], axis=1)
    final_df.to_csv('dummy_datasets/'+dataset_name+'.csv', index=False)
    return final_df


def generate_cont_data(no_of_samples, features):
    cont_data = pd.DataFrame()
    for feature in features:
        start, end, avg, std_dev = ask_for_data_range(feature)
        cont_data[feature] = np.round(np.clip(np.random.normal(loc=avg, scale=std_dev, size=no_of_samples), start, end))

    return cont_data


def ask_for_data_range(feature):
    st.write('For', feature, ':')
    start, end = map(float, st.text_input('Enter a range for data (e.g. 10, 50)', key=feature).split(','))
    avg, std_dev = map(float, st.text_input('Enter mean and standard deviation (e.g. 30, 50)', key=feature+' ').split(','))
    return start, end, avg, std_dev


def generate_disc_data(no_of_samples, features):
    disc_data = pd.DataFrame()
    for feature in features:
        types = ask_for_types(feature)
        disc_data[feature] = [np.random.choice(types) for _ in range(no_of_samples)]
    return disc_data


def ask_for_types(feature):
    st.write('For', feature, ":")
    lst = st.text_input('Enter distinct values (e.g. Male, Female, Transgender)', key=feature).split(',')
    return lst
